import sys

from peng_utils import contact_img, get_image

sys.path.append("./peng_utils")
import argparse
import os

import torch
import numpy as np

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from decord import VideoReader, cpu

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_video(video_path, fps=1):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames



class TestLLaMAVID:
    def __init__(self, device=None) -> None:
        model_path = "/home/fx/Exp2/test/EmbodiedEval/work_dirs/llama-vid/llama-vid-7b-full-224-video-fps-1"
        model_base = None
        load_8bit = False
        load_4bit = False
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.top_p = 0.7
        self.device = device

        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path,
                                                                                                   model_base,
                                                                                                   model_name,
                                                                                                   load_8bit,
                                                                                                   load_4bit)
        self.model_name = model_name
        if 'llama-2' in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower() or "vid" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if device is not None:
            self.move_to_device(device)

        # TODO: check model.to()

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        # self.model.to(device=self.device, dtype=self.dtype)

    def __device__(self):
        return self.device

    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):
        outputs = []
        conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        image_list = contact_img(image_list_list)
        for image, inp in zip(image_list, question_list):
            image = get_image(image)
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
            conv = conv_templates[self.conv_mode].copy()

            self.model.update_prompt([[inp]])

            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

            outputs.append(output)

        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")

        return outputs

    ''' TODO
    # files list -> files list list
    # prompt -> prompt list
    '''

    @torch.no_grad()
    def generate_clip(self, files_list, question_list):
        outputs = []
        conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        for file, inp in zip(files_list, question_list):
            image = load_video(file)
            # print(f"video: {type(image)} {len(image)}")
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
            image_tensor = [image_tensor]
            conv = conv_templates[self.conv_mode].copy()

            self.model.update_prompt([[inp]])

            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

            outputs.append(output)

        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")

        return outputs
