from transformers import LlavaForConditionalGeneration, LlavaNextForConditionalGeneration
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, LlavaNextMultiModalProjector
from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image, image_size_to_num_patches
from dataclasses import dataclass
from transformers import AutoModel
import torch
from typing import Optional, Union, Tuple, List
import torch.nn as nn
from .config_llavanpr import LlavaWithResNetConfig, LlavaWithVisionExpertConfig, ResnetConfig
# from models.config_llava_new import LlavaWithResNetConfig, LlavaWithVisionExpertConfig, ResnetConfig
from .resnet import resnet50
import timm
import torch.nn.functional as F
from functorch import vmap
from transformers import PreTrainedModel
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum
import copy
import numpy as np

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


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
        x = x.squeeze(0)                         # [B,N2,D]

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
        dtype = x.dtype
        x = x.to(torch.float32)
        norm = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x * norm 
        out = self.scale * x.to(dtype)
        return out

def draw_attn(attn, npr_len):
    weight = attn.mean(1).squeeze().mean(0)
    npr = weight[:npr_len]
    clip = weight[npr_len:]
    npr_weight = npr.mean()
    clip_weight = clip.mean()
    return npr, clip
    

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
        # [B, 1, 1, 1, L_kv] 以便广播到 sim
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
        if self.layer_idx == 0 and is_draw:
            npr, clip = draw_attn(attn, x.shape[2])
            return npr, clip
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)

        if q_mask is not None:
            out = out * q_mask.view(out.size(0), 1, -1, 1)

        #out = out.squeeze(0)
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

        return kl_qp.clamp(min=0.0)

    def forward(self, npr_features, clip_features, clip_logits, npr_logits):
        """
        npr_features: (seq_len, dim)
        clip_features: (seq_len, dim)
        """
        npr_len = npr_features.shape[0]
        clip_len = clip_features.shape[0]
        npr_features_b = npr_features.unsqueeze(0)
        clip_features_b = clip_features.unsqueeze(0)
        # npr -> clip cost matrix
        cost_npr2clip = cost_npr2clip = self.pairwise_symmetric_js_from_logits(npr_logits, clip_logits, eps=1e-8)
        #cost_npr2clip = self.pairwise_symmetric_kl_from_logits(npr_logits, clip_logits, eps=1e-8)
        cost_npr2clip_for_npr_source = cost_npr2clip.transpose(1,2)
        cost_npr2clip_for_npr_source = cost_npr2clip_for_npr_source / (cost_npr2clip_for_npr_source.max().detach() + 1e-8)

        # compute gamma_n2c
        gamma_n2c = self.forward_cost_to_plan(cost=cost_npr2clip_for_npr_source)
        transported_npr2clip = torch.bmm(gamma_n2c, npr_features_b)
        transported_npr2clip = transported_npr2clip.squeeze(0) # (clip_seq_len, dim)


        return transported_npr2clip, cost_npr2clip_for_npr_source.mean()


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
        text_embedding = torch.load(r'./text_embedding.pth')['features']
        text_embedding = text_embedding / _get_vector_norm(text_embedding)
        self.text_embedding = torch.nn.Parameter(text_embedding, requires_grad=False)
        self.ot_op = OptimalTransport(epsilon, max_iter, threshold)
        self.shared_proj_clip = nn.Linear(inner_dim, inner_dim)
        ## Cross attention
        self.cross_attn = PerceiverResampler(input_dim, inner_dim, config.num_resampler_layers)
        

    def cal_logit_clip(self, clip_feat):
        image_embeds = clip_feat / _get_vector_norm(clip_feat)
        # cosine similarity as logits
        logits_per_text = torch.matmul(self.text_embedding.detach(), image_embeds.t().to(self.text_embedding.device))
        logits_per_image = logits_per_text.t()
        logits_per_image = logits_per_image.sigmoid().unsqueeze(0)
        return logits_per_image


    def forward(self, npr_feat, clip_feat, teacher_logit):
        b = npr_feat.shape[0]
        npr_feat_fusioned_list = []
        clip_feat_fusioned_list = []
        attention_weight_list = []
        #aux_semantic = 0.
        aux_kl = torch.tensor(0.)
        for idx in range(b):
            npr_feat_single = npr_feat[idx]
            clip_feat_single = clip_feat[idx]
            patch_num, seq_len, dim = clip_feat_single.shape
            clip_feat_single = clip_feat_single.view(-1, dim)

            ## Cross attn
            npr_feat_single_cross = self.cross_attn(npr_feat_single.unsqueeze(0), clip_feat_single.unsqueeze(0))
            

            ## OT Fusion
            npr_feat_single = self.npr_proj_in(npr_feat_single)
            clip_feat_single = self.clip_proj_in(clip_feat_single)
            clip_logits = self.head_clip(clip_feat_single) # .unsqueeze(0)
            clip_logits = self.cal_logit_clip(clip_logits)
            clip_logits = torch.cat([clip_logits, 1.- clip_logits], dim=-1) # [Fake, Real]
            teacher_logit_cur = teacher_logit[idx].sigmoid().unsqueeze(0)
            teacher_logit_cur = torch.cat([teacher_logit_cur, 1. - teacher_logit_cur], dim=-1)
            npr2clip, kl = self.ot_op(npr_feat_single, clip_feat_single, clip_logits, teacher_logit_cur)
            npr2clip = self.shared_proj_clip(npr2clip)
            clip_feat_single_fusion = clip_feat_single + npr2clip
            # auxilariy losses for monitor
            aux_kl += kl.detach().cpu()
            

            # fusion
            npr_feat_fusioned = self.npr_proj_out(npr_feat_single_cross)
            clip_feat_fusioned = self.clip_proj_out(clip_feat_single_fusion)
            clip_feat_fusioned = clip_feat_fusioned.view(patch_num, seq_len, dim)

            npr_feat_fusioned_list.append(npr_feat_fusioned)
            clip_feat_fusioned_list.append(clip_feat_fusioned)
        
        aux_kl = aux_kl / b

        return clip_feat_fusioned_list, npr_feat_fusioned_list, attention_weight_list, aux_kl



class OrthogonalLoss(nn.Module):
    def __init__(self, normalize=True, eps=1e-6):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, zA, zB):
        """
        zA, zB: (batch_size, dim)
        return: scalar loss
        """

        if self.normalize:
            zA = (zA - zA.mean(0)) / (zA.std(0) + self.eps)
            zB = (zB - zB.mean(0)) / (zB.std(0) + self.eps)


        B = zA.size(0)
        C = (zA.T @ zB) / B

        loss = torch.mean(C ** 2)

        return loss

@dataclass
class CustomLlavaNextCausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    ar_loss: Optional[torch.FloatTensor] = None
    cos_loss: Optional[torch.FloatTensor] = None
    cos: Optional[torch.FloatTensor] = None
    orth_loss: Optional[torch.FloatTensor] = None
    kl: Optional[torch.FloatTensor] = None


def normalize_feat(feat, epsilon=1e-10):
    norms = torch.linalg.norm(feat, dim=-1, keepdim=True)
    norm_feats = feat / (norms + epsilon)
    return norm_feats

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
        if config.pretrain_path != "":
            self.pretrained_weights = torch.load(config.pretrain_path, map_location='cpu')

            self.model.load_state_dict(self.pretrained_weights, strict=False)

    def forward(self, tensor):
        return self.model.forward_features(tensor)

    def forward_logit(self, tensor):
        return self.model(tensor)

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
    

class CustomLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    config_class = LlavaWithVisionExpertConfig
    

    def __init__(self, config: LlavaWithVisionExpertConfig):
        super().__init__(config)
        self.vision_tower_expert = AutoModel.from_config(config.expert_config)
        expert_projector_config = copy.deepcopy(config)
        expert_projector_config.vision_config.hidden_size = 512
        self.multi_modal_expert_projector = LlavaNextMultiModalProjector(expert_projector_config)
        resampler_config = ResamplerConfig()
        resampler_config.hidden_size = config.text_config.hidden_size
        resampler_config.vision_hidden_size = 512
        self.multi_modal_expert_projector_otfusion = OTFusion(resampler_config)
        

    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, add_image_features=None, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_select_strategy (`str`)
                The feature selection strategy used to select the vision feature from the vision backbone.
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        # num_patch = image_features.shape[0]
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                if vision_feature_select_strategy == "default":
                    expected_num_patches = height * width
                elif vision_feature_select_strategy == "full":
                    expected_num_patches = height * width + 1
                elif vision_feature_select_strategy == "expert":
                    expected_num_patches = height * width
                if expected_num_patches != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            if add_image_features is not None:
                new_image_features.append(torch.cat((add_image_features[image_idx].reshape((-1, image_feature.shape[-1])), image_feature), dim=0))
                feature_lens.append(add_image_features[image_idx].reshape((-1, image_feature.shape[-1])).size(0) + image_feature.size(0))
            else:
                new_image_features.append(image_feature)
                feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

    def get_clip_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        **kwargs,
    ):
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        elif vision_feature_select_strategy == "expert":
            selected_image_feature = selected_image_feature[:, 1:]

        return selected_image_feature

    def get_expert_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        npr_image_tensor=None,
        **kwargs,
    ):

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        weight_type = self.vision_tower_expert.dtype
        # add_image_features = self.vision_tower_expert(F.interpolate(pixel_values, size=(224, 224), mode='bilinear').to(dtype=weight_type)) # 完整图像放在最前面
        add_image_features = self.vision_tower_expert(npr_image_tensor.to(dtype=weight_type))
        #add_image_features_fc = self.multi_modal_expert_projector(add_image_features)
        return add_image_features

    def get_clip_npr_final_features(
            self,
            pixel_values: torch.FloatTensor,
            image_sizes: torch.Tensor,
            vision_feature_layer=None,
            vision_feature_select_strategy=None,
            npr_image_tensor = None,
            is_draw = False,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_layer (`int`):
                The index of the layer to select the vision feature.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (List[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """
        # ! infer image_num_patches from image_sizes
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        elif vision_feature_select_strategy == "expert":
            selected_image_feature = selected_image_feature[:, 1:]

            weight_type = self.vision_tower_expert.dtype
            add_image_features = self.vision_tower_expert(npr_image_tensor.to(dtype=weight_type))
            
            add_image_features = add_image_features.to(selected_image_feature.device, selected_image_feature.dtype)

            

        latent_image_feature = self.multi_modal_expert_projector(add_image_features, selected_image_feature, image_num_patches, is_draw=is_draw)
        if is_draw:
            return selected_image_feature
        
        return latent_image_feature, add_image_features, selected_image_feature

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            image_sizes: torch.Tensor,
            vision_feature_layer: int,
            vision_feature_select_strategy: str,
            npr_image_tensor = None,
            is_draw = False,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_layer (`int`):
                The index of the layer to select the vision feature.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (List[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        elif vision_feature_select_strategy == "expert":
            selected_image_feature = selected_image_feature[:, 1:]

            weight_type = self.vision_tower_expert.dtype
            add_image_features = self.vision_tower_expert(npr_image_tensor.to(dtype=weight_type))
            add_image_features_logit = self.vision_tower_expert.forward_logit(npr_image_tensor.to(dtype=weight_type))
            add_image_features = add_image_features.to(selected_image_feature.device, selected_image_feature.dtype)

            
        image_features = self.multi_modal_projector(selected_image_feature)
        add_image_features = self.multi_modal_expert_projector(add_image_features)

        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, add_image_features, attention_weight, aux_kl = self.multi_modal_expert_projector_otfusion(add_image_features, image_features, add_image_features_logit)

        return image_features, add_image_features, aux_kl

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
        image_token_index=None,
        ignore_index=-100,
    ):
        """
        Merge input_ids with with image features into final embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`):
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                :abels need to be recalculated to support training (if provided)
            image_token_index (`int`, *optional*)
                Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
            ignore_index (`int`, *optional*)
                Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (8)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but image token sizes are different, then cannot infer left or right padding
                ```python
                cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
                chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)
                prompts = [
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                ]
                inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                    chart_img has 2634 tokens, while cat_img has 2340 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
        ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

        with torch.no_grad():
            # ! in llava 1.6, number of patches is variable
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = True if not self.training else False
            if batch_size > 1 and not self.training:
                if _left_padding and not _right_padding:
                    left_padding = True
                elif not _left_padding and _right_padding:
                    left_padding = False
                elif not _left_padding and not _right_padding:
                    # both side is 1, so cannot tell
                    left_padding = self.padding_side == "left"
                else:
                    # invalid attention_mask
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens = feature_lens.to(input_ids.device)
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_embed_dim), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                max_embed_dim = max_embed_dim.to(target_device)
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids, final_labels, final_input_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        logits_to_keep: int=0,
        npr_image_tensor = None,
        is_draw = False,
        **lm_kwargs,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast, CustomLlavaNextCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            # In case we're in decoding stage, legacy behavior is checked by presence of pixel values even if use_cache=True
            legacy_processing = (
                (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
            ) or (input_ids.shape[-1] == 1 and pixel_values is not None)

        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features, add_image_features, aux_kl = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                npr_image_tensor=npr_image_tensor,
                is_draw=is_draw,
            )
            if is_draw:
                return image_features

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                add_image_features=add_image_features,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=self.image_newline,
            )



        if legacy_processing:
            # logger.warning_once(
            #     "Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. "
            #     "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
            #     "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
            #     "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            # )
            if input_ids.shape[1] != 1:
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

                inputs_embeds, attention_mask, position_ids, labels, _ = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            else:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        # TODO: @raushan retain only the new behavior after v4.47
        elif image_features is not None:
            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            weight_type = inputs_embeds.dtype
            image_features = image_features.to(dtype=weight_type)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        ar_loss = None
        cos_loss = None
        cos = None
        orth_loss = None
        kl = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            ar_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )


            loss = ar_loss
            if cos_loss is not None:
                loss += cos_loss
            if orth_loss is not None:
                #loss += (0.5 * orth_loss)
                loss += orth_loss
            
            # if aux_semantic is not None:
            #     cos = aux_semantic
            if aux_kl is not None:
                kl = aux_kl
            

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CustomLlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            ar_loss=ar_loss,
            cos_loss=cos_loss,
            orth_loss=orth_loss,
            cos=cos,
            kl=kl,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    config_class = LlavaWithResNetConfig
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
        image_token_index=None,
        ignore_index=-100,
    ):
        """
        Merge input_ids with with image features into final embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`):
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                :abels need to be recalculated to support training (if provided)
            image_token_index (`int`, *optional*)
                Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
            ignore_index (`int`, *optional*)
                Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (8)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but image token sizes are different, then cannot infer left or right padding
                ```python
                cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
                chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)
                prompts = [
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                ]
                inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                    chart_img has 2634 tokens, while cat_img has 2340 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
        ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

        with torch.no_grad():
            # ! in llava 1.6, number of patches is variable
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = True if not self.training else False
            if batch_size > 1 and not self.training:
                if _left_padding and not _right_padding:
                    left_padding = True
                elif not _left_padding and _right_padding:
                    left_padding = False
                elif not _left_padding and not _right_padding:
                    # both side is 1, so cannot tell
                    left_padding = self.padding_side == "left"
                else:
                    # invalid attention_mask
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens = feature_lens.to(input_ids.device)
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_embed_dim), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                max_embed_dim = max_embed_dim.to(target_device)
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids, final_labels, final_input_ids
    # @add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            vision_feature_layer: Optional[int] = None,
            vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            # In case we're in decoding stage, legacy behavior is checked by presence of pixel values even if use_cache=True
            legacy_processing = (
                                        (input_ids == self.config.image_token_index).sum(
                                            1).max() < self.config.image_seq_length
                                ) or (input_ids.shape[-1] == 1 and pixel_values is not None)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        if legacy_processing:
            # prefill stage vs decoding stage (legacy behavior copied)
            if input_ids.shape[1] != 1:
                inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            else:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        # TODO: @raushan retain only the new behavior after v4.47
        elif image_features is not None:
            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0] * image_features.shape[1]

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0][0]
        hidden_states = outputs[1]
        outputs = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1):].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        ), hidden_states

AutoModel.register(ResnetConfig, ResnetExpertModel)
# AutoModel.register(ResnetConfig, ResnetExpertModel)


from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoModel.register(ResnetConfig, ResnetExpertModel)
AutoModel.register(LlavaWithVisionExpertConfig, CustomLlavaNextForConditionalGeneration)
