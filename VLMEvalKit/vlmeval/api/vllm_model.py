from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from io import BytesIO
import time

# use for qwen2_5 vl
def ensure_image_url(image) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if not isinstance(image,str):
        image = "data:image/jpeg;base64," + encode_image_to_base64(image)
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        image = Image.open(image)
        image = "data:image/jpeg;base64," + encode_image_to_base64(image)
        return image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def process_video_qwen2_5vl(video_url, use_cache=True):

    cache_path = r'./cache'
    file_name = os.path.basename(os.path.basename(video_url)) + '.json'
    cache_file_path = os.path.join(cache_path, file_name)
    if os.path.exists(cache_file_path) and use_cache:
        video_base64_frames = load(cache_file_path)['base64']
        return video_base64_frames

    video_message = [
        {'content': [{
            "type": "video",
            "video": video_url,
            "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
            'fps': 3.0  # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
        }]}
    ]
    _, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
    assert video_inputs is not None, "video_inputs should not be None"
    video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    base64_frames = []
    for frame in video_input:
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        base64_frames.append(base64_str)
    
    video_base64_frames = f"data:video/jpeg;base64,{','.join(base64_frames)}"
    dic_cache = {"base64": video_base64_frames}
    dump(dic_cache, cache_file_path)
    return video_base64_frames


class VLLMWrapper(BaseAPI):

    is_api: bool = True
    VIDEO_LLM: bool = True
    def __init__(self,
                 model: str = 'qwen2_5',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 top_p=0.8,
                 top_k=1,
                 repetition_penalty=1.0,
                 system_prompt: str = "",
                 max_tokens: int = 1024,
                 min_pixels=1280 * 28 * 28,
                 max_pixels=16384 * 28 * 28,
                 api_base = None,
                 api_key = None,
                 **kwargs,
                 ):
        
        self.model = model
        if api_base is None:
            self.api_base = os.environ.get('VLLM_API_BASE', None)
        else:
            self.api_base = api_base
        if api_key is None:
            self.api_key = os.environ.get('VLLM_API_KEY',"None")
        else:
            self.api_key = api_key
        assert self.api_base, f"VLLM_API_BASE is {self.api_base}"
        self.fail_msg = 'Failed to obtain answer via API.'
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.top_p = top_p
        self.generate_kwargs = dict(
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature
        )
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def _prepare_content(self, inputs: list[dict[str, str]], dataset=None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image_url', 'image_url': {"url":ensure_image_url(s['value'])}}
                if self.model == 'qwen2_5':
                    if dataset == 'OCRBench':
                        item['min_pixels'] = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                        if self.max_pixels is not None:
                            item['max_pixels'] = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            item['min_pixels'] = self.min_pixels
                        if self.max_pixels is not None:
                            item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                #TEMPLATE_PROMPT = """{}\nRespond only with "Real" or "Fake" only."""
                #prompt = "Your sole purpose is to determine if a given image is real or fake. Respond only with 'Real' or 'Fake' only."
                #prompt = TEMPLATE_PROMPT.format(s['value'])
                #print(prompt)
                prompt = s['value']
                item = {'type': 'text', 'text': prompt}
            elif s['type'] == 'video':
                if self.model == 'qwen2_5':
                    if self.verbose:
                        video_path = s['value']
                        print(f"{video_path}")
                    start_time = time.time()
                    video_base64_frames = process_video_qwen2_5vl(s['value'])
                    print("Process video cost time: {} min".format((time.time() - start_time) / 60))
                    item = {'type': "video_url", "video_url": {"url": video_base64_frames}}
                else:
                    raise ValueError("Not Supported video model: {}".format(self.model))
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, inputs, **kwargs):
        
        start_time = time.time()
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append(
            {'role': 'user', 'content': self._prepare_content(inputs, dataset=kwargs.get('dataset',None))}
        )
        # if self.verbose:
        #     print(f'\033[31m{messages}\033[0m')

        # generate
        generation_kwargs = self.generate_kwargs.copy()
        kwargs.pop('dataset', None)
        generation_kwargs.update(kwargs)
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
            model_name = client.models.list().data[0].id
            generate_start_time = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **generation_kwargs,
            )
            print("generate cost time: {} mins".format((time.time() - generate_start_time) / 60.))
            if self.verbose:
                print(response)
                #response.choices[0].message
            answer = response.choices[0].message.content
            print("Total cost time: {} mins".format((time.time() - start_time) / 60.))
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(f'The input messages are {inputs}.')
            return -1, '', ''
