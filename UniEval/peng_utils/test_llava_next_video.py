import os
import re
from typing import Dict

from decord import VideoReader, cpu

os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys

import numpy as np

from peng_utils import contact_img, get_image

sys.path.append("./peng_utils")
import torch

from llavanext.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llavanext.conversation import conv_templates, SeparatorStyle
from llavanext.model.builder import load_pretrained_model
from llavanext.utils import disable_torch_init
from llavanext.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import transformers
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

class TestLLaVANextVideo:
    def __init__(self, device=None) -> None:
        disable_torch_init()
        model_path = "/home/fx/Exp2/video_model/LLaVA-NeXT/MODELS/LLaVA-NeXT-Video-7B-Qwen2" # "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"
        model_base = None
        load_8bit = True
        load_4bit = False
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.model_name = model_name
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base,
                                                                               model_name, load_8bit,
                                                                               load_4bit)
        self.model.eval()

        if "llama-2" in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        self.conv_mode = "qwen_1_5"
        self.device = device


        if device is not None:
            self.move_to_device(device)


    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            device = device
        else:
            self.dtype = torch.float16
            device = 'cpu'
        # vision_tower = self.model.get_model().vision_tower[0]
        # vision_tower.to(device=device, dtype=self.dtype)
        # self.model.to(device=device, dtype=self.dtype)


    def __device__(self):
        return self.device

    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):
        outputs = []
        # image_list = contact_img(image_list_list)
        for image_list, inp in zip(image_list_list, question_list):

            conv = conv_templates[self.conv_mode].copy()
            image_tensors = []
            # special_token = []
            for image in image_list:
                image = get_image(np.array(image))
                image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
                # special_token += [DEFAULT_IMAGE_TOKEN]
                image_tensors.append(image_tensor)

            # conv.append_message(conv.roles[0], inp)
            # conv.append_message(conv.roles[1], None)
            # # conv.append_message(conv.roles[1], None)
            # prompt = conv.get_prompt()
            real_prompt = '<|im_start|>usr\n<image><|im_end|>'+inp
            print(f"real prompt:{real_prompt}")
            sources = {'value':real_prompt, 'from': 'human'}
            input_ids = preprocess_qwen([sources,{'from': 'gpt','value': None}], self.tokenizer, has_image=True).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, images=image_tensors, do_sample=True, temperature=0.2,
                                            max_new_tokens=1024, use_cache=True, top_p=None, num_beams=1,)
                                            # streamer=streamer, stopping_criteria=[stopping_criteria])

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            outputs.append(output)

        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")

        return outputs


    @torch.no_grad()
    def generate_clip(self, files_list, question_list, num_segments=None, img_list=None, num_beams=1, temperature=1.0):
        outputs = []

        for files, question in zip(files_list, question_list):
            print(f"question_list:{question_list} inp:{question}")
            conv = conv_templates[self.conv_mode].copy()
            tensor = []
            video_ext = ['.mp4', '.mov', '.mkv', '.avi']
            files = files if isinstance(files, list) else [files]
            for file in files:
                if os.path.splitext(file)[-1].lower() in video_ext:
                    video, frame_time, video_time = load_video(file, 16, 1, force_sample=True)
                    file = self.image_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].cuda().half()
                else:
                    raise ValueError(
                        f'Support video of {video_ext}, but found {os.path.splitext(file)[-1].lower()}')
                print(file.shape)
                tensor.append(file)

            question = DEFAULT_IMAGE_TOKEN + "\n" + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            print(f"real prompt:{prompt_question}")
            sources = {'value': prompt_question, 'from': 'human'}
            input_ids = preprocess_qwen([sources, {'from': 'gpt', 'value': None}], self.tokenizer,
                                        has_image=True).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids,
                                                images=tensor,
                                                modalities="video",
                                                do_sample=True,
                                                temperature=0.2,
                                                max_new_tokens=4096,
                                                 use_cache=True,
                                                 top_p=None,
                                                 num_beams=1,
                                            )
                # streamer=streamer, stopping_criteria=[stopping_criteria])

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            outputs.append(output)

        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")

        return outputs