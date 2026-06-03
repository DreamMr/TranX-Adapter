from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel, AutoModel, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast, Qwen3VLCausalLMOutputWithPast
try:
    from .config_qwen3vlnpr import Qwen3VLWithVisionExpertConfig, ResnetConfig
except ImportError:
    from config_qwen3vlnpr import Qwen3VLWithVisionExpertConfig, ResnetConfig
import copy
from torch import nn
from transformers.activations import ACT2FN
import torch
try:
    from .resnet_qwen3vl import resnet50
except ImportError:
    from resnet_qwen3vl import resnet50
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum
import copy
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
import torch.nn.functional as F

class ResnetExpertModel(PreTrainedModel):
    config_class = ResnetConfig
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def __init__(self, config: ResnetConfig):
        super().__init__(config)
        self.model = resnet50(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            use_low_level=config.use_low_level
        )
        #self.model = timm.create_model('resnet50_clip.openai', pretrained=False)
        if config.pretrain_path != "":
            self.pretrained_weights = torch.load(config.pretrain_path, map_location='cpu')

            self.model.load_state_dict(self.pretrained_weights, strict=False)

    def forward(self, tensor):
        return self.model.forward_features(tensor)
        #return self.model(tensor)
        
    def forward_logit(self, tensor):
        return self.model(tensor)


class Qwen3VLMultiModalExpertProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.out_hidden_size = config.out_hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.out_hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


@dataclass
class ResamplerConfig:
    hidden_size = None
    num_fusion_layers= 6
    num_resampler_layers = 3 # default is 3
    vision_hidden_size = None
    epsilon = 0.1
    max_iter = 1000
    threshold = 1e-9
    inner_dim = 512

class PerceiverResampler(nn.Module):
    def __init__(self, input_dim, hidden_size, num_resampler_layers):
        super().__init__()
        dim = hidden_size
        depth=num_resampler_layers
        self.layers = nn.ModuleList([])
        self.linear_x = nn.Linear(input_dim, hidden_size)
        self.linear_latent = nn.Linear(input_dim, hidden_size)
        for layer_idx in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=64, heads=8, layer_idx=layer_idx),
                        FeedForward(dim=dim, mult=4),
                    ]
                )
            )

        self.norm = RMSNorm(dim)


    def forward(self, x_s, latents_s, is_draw=False):
        device = x_s.device
        N_img = x_s.shape[1]
        N_lat = latents_s.shape[1]

        x = self.linear_x(x_s)                         # [B, N_img, D]
        latents = self.linear_latent(latents_s)       # [sumP, N_lat, D]

        
        
        for attn, ff in self.layers:
            if is_draw:
                return attn(x, latents, kv_mask=None, q_mask=None, is_draw=is_draw)
            x = attn(x, latents, kv_mask=None, q_mask=None) + x
            x = ff(x) + x

        x = self.norm(x)                         # [B,1,N2,D]

        return x
        


# =================================resampler related =================================
def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 计算均方根
        dtype = x.dtype
        x = x.to(torch.float32)
        norm = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x * norm 
        out = self.scale * x.to(dtype)
        return out


class EncoderAttention(nn.Module):
    def __init__(self, input_dim):
        super(EncoderAttention, self).__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.W_npr = nn.Sequential(
            nn.RMSNorm(input_dim),
            nn.Linear(input_dim, input_dim)
        )
        self.W_clip = nn.Sequential(
            nn.RMSNorm(input_dim),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, npr_feat, clip_feat):
        npr_feat_flat = torch.mean(npr_feat, dim=1)
        clip_feat_flat = torch.mean(clip_feat, dim=1)

        combined_features = torch.cat([npr_feat_flat, clip_feat_flat], dim=1)

        attention_score = self.attention_net(combined_features)
        attention_weight = F.softmax(attention_score, dim=1)
        w_npr = attention_weight[:, 0].unsqueeze(1).unsqueeze(2)
        w_clip = attention_weight[:, 1].unsqueeze(1).unsqueeze(2)

        final_npr = (1. + w_npr) * npr_feat
        final_clip = (1. + w_clip) * clip_feat

        output_npr = self.W_npr(final_npr)
        output_clip = self.W_clip(final_clip)

        return output_npr, output_clip, attention_weight

class EncoderFusion(nn.Module):
    def __init__(self, input_dim, layer_depth=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_depth = layer_depth
        for layer_idx in range(layer_depth):
            self.layers.append(
                EncoderAttention(
                    input_dim
                )
            )
        
    def forward(self, npr_feat, clip_feat):
        attention_weight_list = []
        npr_feat_tmp = npr_feat
        clip_feat_tmp = clip_feat
        for idx in range(self.layer_depth):
            npr_feat_tmp, clip_feat_tmp, attn_weight = self.layers[idx](npr_feat_tmp, clip_feat_tmp)
            attention_weight_list.append(attn_weight)
        return npr_feat_tmp, clip_feat_tmp, attention_weight_list

class OptimalTransport(nn.Module):
    def __init__(self, epsilon=0.1, max_iter=1000, threshold=1e-9):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.threshold = threshold

    
    def _log_sinkhorn(self, logK, loga, logb):
        # logK: (B, N, M), loga: (B,N), logb: (B,M)
        B, N, M = logK.shape
        device = logK.device
        logu = torch.zeros((B, N), device=device, dtype=logK.dtype)
        logv = torch.zeros((B, M), device=device, dtype=logK.dtype)

        for i in range(self.max_iter):
            logKv = torch.logsumexp(logK + logv.unsqueeze(1), dim=2)  # (B,N)
            logu = loga - logKv
            logKTu = torch.logsumexp(logK.transpose(1,2) + logu.unsqueeze(1), dim=2)  # (B,M)
            new_logv = logb - logKTu
            if i % 10 == 0:
                if (new_logv - logv).abs().mean() < self.threshold:
                    logv = new_logv
                    break
            logv = new_logv

        logGamma = logu.unsqueeze(2) + logK + logv.unsqueeze(1)
        Gamma = torch.exp(logGamma)
        # re-normalize to ensure numeric safety
        Gamma = Gamma / (Gamma.sum(dim=(1,2), keepdim=True) + 1e-30)
        return Gamma

    def forward_cost_to_plan(self, cost, a=None, b=None):
        """
        cost: (B, N, M) or (1,N,M)
        a: (B, N) or None -> uniform
        b: (B, M) or None -> uniform
        returns gamma (B,N,M)
        """
        if cost.dim() == 2:
            cost = cost.unsqueeze(0)
        B, N, M = cost.shape
        device = cost.device
        if a is None:
            a = torch.full((B, N), 1.0 / N, device=device, dtype=cost.dtype)
        if b is None:
            b = torch.full((B, M), 1.0 / M, device=device, dtype=cost.dtype)
        # logK = -cost / eps
        logK = -cost / float(self.epsilon)
        # logs of marginals
        loga = torch.log(a + 1e-30)
        logb = torch.log(b + 1e-30)
        gamma = self._log_sinkhorn(logK, loga, logb)
        return gamma

    def pairwise_cos_cost(self, src, tgt):
        # src: (1,N,D) or (B,N,D), tgt: (1,M,D) or (B,M,D)
        # returns cost (B,N,M): cost = 1 - cosine_similarity
        if src.dim() == 2:
            src = src.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
        s = F.normalize(src, dim=-1)
        t = F.normalize(tgt, dim=-1)
        cost = 1.0 - torch.bmm(s, t.transpose(1,2))
        return cost
    
    def pairwise_cosine(self, npr_feat, clip_feat):
        """
        npr_feat: torch.tensor shape: [batch_size, seq_len1, dim]
        clip_feat: torch.tensor, shape: [batch_size, seq_len2, dim]
        """
        # 1. L2 归一化 (沿着特征维度 dim=-1)
        # F.normalize 会自动处理除以 0 的情况 (默认 eps=1e-12)
        npr_feat_norm = F.normalize(npr_feat, p=2, dim=-1)
        clip_feat_norm = F.normalize(clip_feat, p=2, dim=-1)

        # 2. 矩阵乘法
        # [B, L1, D] @ [B, D, L2] -> [B, L1, L2]
        # 需要将 clip_feat_norm 转置最后两个维度
        scores = torch.matmul(npr_feat_norm, clip_feat_norm.transpose(1, 2))
        return scores

    def pairwise_symmetric_js_from_logits(self, logits_p, logits_q, eps=1e-12, temp=1.0):
        """
        logits_p: (B, N, C) NPR
        logits_q: (B, M, C) CLIP
        returns cost matrix (B, N, M) where element (i,j) is JS(p_i || q_j)
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
        """
        if logits_p.dim() == 2:
            logits_p = logits_p.unsqueeze(0)
            logits_q = logits_q.unsqueeze(0)
    
        # (B, N, C)
        p = F.softmax(logits_p / float(temp), dim=-1).clamp(min=eps)
        # (B, M, C)
        q = F.softmax(logits_q / float(temp), dim=-1).clamp(min=eps)

        
        # p_exp: (B, N, 1, C)
        p_exp = p.unsqueeze(2)
        # q_exp: (B, 1, M, C)
        q_exp = q.unsqueeze(1)
        
        m = 0.5 * (p_exp + q_exp)

        
        log_p = torch.log(p_exp)
        log_q = torch.log(q_exp)
        log_m = torch.log(m.clamp(min=eps))

        
        # KL(P||M) = sum(P * (logP - logM))
        kl_p_m = (p_exp * (log_p - log_m)).sum(dim=-1)
        
        # KL(Q||M) = sum(Q * (logQ - logM))
        kl_q_m = (q_exp * (log_q - log_m)).sum(dim=-1)

        js_div = 0.5 * (kl_p_m + kl_q_m)

        return js_div.clamp(min=0.0)

    def pairwise_symmetric_kl_from_logits(self, logits_p, logits_q, eps=1e-12, temp=1.0):
        """
        logits_p: (B, N, C) NPR
        logits_q: (B, M, C) CLIP
        returns cost matrix (B, N, M) where element (i,j)= 0.5*(KL(p_i||q_j)+KL(q_j||p_i))
        KL(p||q) = sum_k p_k (log p_k - log q_k)
        We'll compute pairwise efficiently with broadcasting.
        """
        if logits_p.dim() == 2:
            logits_p = logits_p.unsqueeze(0)
            logits_q = logits_q.unsqueeze(0)
        B, N, C = logits_p.shape
        #print(logits_q.shape)
        if len(logits_q.shape) == 4:
            debug = 0
        _, M, _ = logits_q.shape

        # softmax probabilities with temperature
        p = F.softmax(logits_p / float(temp), dim=-1).clamp(min=eps)  # (B,N,C)
        q = F.softmax(logits_q / float(temp), dim=-1).clamp(min=eps)  # (B,M,C)

        # log probs
        logp = torch.log(p)
        logq = torch.log(q)

        # compute KL(p_i || q_j) = sum_k p_i(k) * (logp_i(k) - logq_j(k))
        # we want matrix of shape (B, N, M)
        # Expand and use broadcasting:
        # p: (B,N,1,C), logp: (B,N,1,C), logq: (B,1,M,C)
        # p_exp = p.unsqueeze(2)      # (B,N,1,C)
        # logp_exp = logp.unsqueeze(2)
        # logq_exp = logq.unsqueeze(1) # (B,1,M,C)
        # kl_pq = (p_exp * (logp_exp - logq_exp)).sum(dim=-1)  # (B,N,M)

        # similarly KL(q_j || p_i): expand q and p accordingly
        q_exp = q.unsqueeze(1)       # (B,1,M,C)
        logq_exp2 = logq.unsqueeze(1)
        logp_exp2 = logp.unsqueeze(2) # (B,N,1,C)
        kl_qp = (q_exp * (logq_exp2 - logp_exp2)).sum(dim=-1)  # (B,N,M) but transposed semantics

        # symmetric:
        return kl_qp.clamp(min=0.0)

    def forward(self, npr_features, clip_features, clip_logits, npr_logits):
        """
        npr_features: (seq_len, dim)
        clip_features: (seq_len, dim)
        """
        npr_len = npr_features.shape[0]
        clip_len = clip_features.shape[0]
        npr_features_b = npr_features
        cost_npr2clip = self.pairwise_symmetric_js_from_logits(npr_logits, clip_logits)
        cost_npr2clip_for_npr_source = cost_npr2clip.transpose(1,2)
        cost_npr2clip_for_npr_source = cost_npr2clip_for_npr_source / (cost_npr2clip_for_npr_source.max().detach() + 1e-8)

        # compute gamma_n2c
        gamma_n2c = self.forward_cost_to_plan(cost=cost_npr2clip_for_npr_source)
        transported_npr2clip = torch.bmm(gamma_n2c, npr_features_b)
        
        return transported_npr2clip, cost_npr2clip_for_npr_source.mean()



class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, layer_idx, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.layer_idx = layer_idx
        inner_dim = dim_head * heads

        self.norm_media = RMSNorm(dim,eps=1e-3)
        self.norm_latents = RMSNorm(dim,eps=1e-3)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    

    @torch.no_grad()
    def _make_broadcast_mask(self, kv_mask, target_shape):
        """
        kv_mask: [B, L_kv] -> broadcast 到 [B, H, T, L_q, L_kv]
        target_shape: sim 的形状 [B, H, T, L_q, L_kv]
        """
        B, H, T, Lq, Lkv = target_shape
        # [B, 1, 1, 1, L_kv] 
        return kv_mask.view(B, 1, 1, 1, Lkv)

    def forward(self, x, latents, kv_mask=None, q_mask=None, is_draw=False):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(x)
        kv_input = latents
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        if kv_mask is not None:
            sim = sim.masked_fill(~self._make_broadcast_mask(kv_mask, sim.shape), float("-inf"))

        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)

        
        if q_mask is not None:
            out = out * q_mask.view(out.size(0), 1, -1, 1)

        return out

class OTFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.hidden_size
        inner_dim = config.inner_dim
        ## optimal transport
        epsilon = config.epsilon
        max_iter = config.max_iter
        threshold=config.threshold
        self.clip_proj_in = nn.Linear(input_dim, inner_dim)
        self.npr_proj_in = nn.Linear(input_dim, inner_dim)
        self.clip_proj_out = nn.Linear(inner_dim, input_dim)
        self.npr_proj_out = nn.Linear(inner_dim, input_dim)
        self.head_clip = nn.Linear(inner_dim, 768)
        
        self.text_embedding = nn.Linear(input_dim, 1)
        self.ot_op = OptimalTransport(epsilon, max_iter, threshold)
        self.shared_proj_clip = nn.Linear(inner_dim, inner_dim)
        ## Cross attention
        self.cross_attn = PerceiverResampler(input_dim, inner_dim, config.num_resampler_layers)
        
        

    def cal_logit_clip(self, clip_feat):
        logits_per_image = self.text_embedding(clip_feat)
        logits_per_image = logits_per_image.sigmoid()
        return logits_per_image


    def forward(self, npr_feat, clip_feat, teacher_logit):
        b = npr_feat.shape[0]
        clip_feat = torch.stack(list(clip_feat), dim=0)
        npr_feat_fusioned_list = []
        clip_feat_fusioned_list = []
        attention_weight_list = []
        aux_kl = torch.tensor(0.)
        npr_feat_single = npr_feat
        clip_feat_single = clip_feat
        batch_size, patch_num, dim = clip_feat_single.shape

        ## Cross attn
        npr_feat_single_cross = self.cross_attn(npr_feat_single, clip_feat_single)

        ## OT Fusion
        clip_logits = self.cal_logit_clip(clip_feat_single)
        clip_logits = torch.cat([clip_logits, 1.- clip_logits], dim=-1) # [Fake, Real]
        npr_feat_single = self.npr_proj_in(npr_feat_single)
        clip_feat_single = self.clip_proj_in(clip_feat_single)
        teacher_logit_cur = teacher_logit.sigmoid()
        teacher_logit_cur = torch.cat([teacher_logit_cur, 1. - teacher_logit_cur], dim=-1)
        npr2clip, kl = self.ot_op(npr_feat_single, clip_feat_single, clip_logits, teacher_logit_cur)
        npr2clip = self.shared_proj_clip(npr2clip)
        clip_feat_single_fusion = clip_feat_single + npr2clip
        # auxilariy losses for monitor
        aux_kl += kl.detach().cpu()

        # fusion
        npr_feat_fusioned = self.npr_proj_out(npr_feat_single_cross)
        clip_feat_fusioned = self.clip_proj_out(clip_feat_single_fusion)
        
        aux_kl = aux_kl
        return clip_feat_fusioned, npr_feat_fusioned, attention_weight_list, aux_kl

@dataclass
class CustomQwen3VLModelOutputWithPast(Qwen3VLModelOutputWithPast):
    kl: Optional[torch.FloatTensor] = None
    
@dataclass
class CustomQwen3VLCausalLMOutputWithPast(Qwen3VLCausalLMOutputWithPast):
    kl: Optional[torch.FloatTensor] = None

class CustomQwen3VLModel(Qwen3VLModel):
    config: Qwen3VLWithVisionExpertConfig
    
    def __init__(self, config: Qwen3VLWithVisionExpertConfig):
        super().__init__(config)
        self.vision_tower_expert = AutoModel.from_config(config.expert_config)
        expert_config = copy.deepcopy(config)
        expert_config.vision_config.hidden_size=512
        self.multi_modal_expert_projector = Qwen3VLMultiModalExpertProjector(expert_config.vision_config)
        
        # OTFusion
        resampler_config = ResamplerConfig()
        resampler_config.hidden_size = config.text_config.hidden_size
        resampler_config.vision_hidden_size = 512
        self.multi_modal_expert_projector_otfusion = OTFusion(resampler_config)
    
    def get_expert_image_features(self, npr_pixel_values):
        weight_type = self.vision_tower_expert.dtype
        add_image_features = self.vision_tower_expert(npr_pixel_values.to(dtype=weight_type))
        add_image_features = self.multi_modal_expert_projector(add_image_features)
        add_image_features_logit = self.vision_tower_expert.forward_logit(npr_pixel_values.to(dtype=weight_type))
        return add_image_features, add_image_features_logit
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        npr_pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        aux_kl = None
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            ### expert image features
            split_sizes = [embed.shape[0] for embed in image_embeds]
            add_image_features, add_image_features_logit = self.get_expert_image_features(npr_pixel_values)
            num_expert = add_image_features.shape[1]
            image_embeds, add_image_features, attention_weight, aux_kl = self.multi_modal_expert_projector_otfusion(add_image_features, image_embeds, add_image_features_logit)
            image_embeds = torch.cat([add_image_features, image_embeds], dim=1).to(inputs_embeds.device, inputs_embeds.dtype)
            
            
            new_deepstack_image_embeds = []
            for layer_embed in deepstack_image_embeds:
                layer_splits = torch.split(layer_embed, split_sizes, dim=0)
                new_layer_feats = []
                for i, feat in enumerate(layer_splits):
                    zeros = torch.zeros((num_expert, feat.shape[1]), dtype=feat.dtype, device=feat.device)
                    new_layer_feats.append(torch.cat((zeros, feat), dim=0))
                new_deepstack_image_embeds.append(torch.cat(new_layer_feats, dim=0))

            deepstack_image_embeds = new_deepstack_image_embeds
            
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return CustomQwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            kl=aux_kl,
        )


class CustomQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    
    config_class = Qwen3VLWithVisionExpertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen3VLModel(config)
        
    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        npr_pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
        ):
        
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        >>> model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            npr_pixel_values=npr_pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return CustomQwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            kl=outputs.kl,
        )

        
        

AutoModel.register(ResnetConfig, ResnetExpertModel)
AutoModel.register(Qwen3VLWithVisionExpertConfig, CustomQwen3VLForConditionalGeneration)