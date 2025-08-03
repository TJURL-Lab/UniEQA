import numpy as np

from . import get_image

import argparse
import os
import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VIDEO_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.serve.utils import load_image, image_ext, video_ext
from videollava.utils import disable_torch_init
from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer




class TestVideoChatGPT:
    def __init__(self, device=None) -> None:
        model_path = "LanguageBind/Video-LLaVA-7B"
        model_base = None
        cache_dir = None
        load_8bit = False # True
        load_4bit = True  # False
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.device = device

        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, processor, self.context_len = load_pretrained_model(model_path, model_base, self.model_name,
                                                                         load_8bit, load_4bit,
                                                                         device=device, cache_dir=cache_dir)
        self.image_processor, self.video_processor = processor['image'], processor['video']
        if 'llama-2' in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if device is not None:
            self.move_to_device(device)


    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            device = device
        else:
            self.dtype = torch.float16
            device = 'cpu'
        # self.model = self.model.to(device, dtype=self.dtype)

    def __device__(self):
        return self.device

    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):
        outputs = []
        for image_list, inp in zip(image_list_list, question_list):
            tensor = []
            special_token = []

            # image_list = image_list_list[0]
            for image in image_list:
                image = Image.fromarray(np.array(image, dtype='uint8')).convert('RGB')
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].to(
                    self.model.device,
                    dtype=torch.float16)
                special_token += [DEFAULT_IMAGE_TOKEN]
                tensor.append(image)

            # for inp in question_list:
            # inp = question_list[0]
            conv = conv_templates[self.conv_mode].copy()
            if image_list_list is not None:
                # first message
                if getattr(self.model.config, "mm_use_im_start_end", False):
                    inp = ''.join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN for i in special_token]) + '\n' + inp
                else:
                    inp = ''.join(special_token) + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                file = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(
                0).to(self.model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            print(f"input_ids:{type(input_ids)} image:{type(image)} ")

            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=tensor,  # video as fake images
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = output
            outputs.append(output)

        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")

        return outputs

    ''' TODO
    # files list -> files list list
    # prompt -> prompt list
    '''
    @torch.no_grad()
    def generate_clip(self, files_list, question_list):
        # print(f"generate_clip: files:{type(files), files}\nquestion:{type(question), question}")
        '''
        2024-08-27 17:02:40 | INFO | stdout | generate_clip: files:(<class 'list'>, ['./benchmarks/EgoTaskQA/videos/29k-22-4-2|P1|2145|3701.mp4'])
        2024-08-27 17:02:40 | INFO | stdout | question:(<class 'list'>, ['What is the precondition of changing the cuttability of watermelon1?'])
        '''
        outputs = []
        for files, question in zip(files_list, question_list):
            conv = conv_templates[self.conv_mode].copy()
            if "mpt" in self.model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            tensor = []
            special_token = []
            files = files if isinstance(files, list) else [files]
            for file in files:
                if os.path.splitext(file)[-1].lower() in image_ext:
                    file = self.image_processor.preprocess(file, return_tensors='pt')['pixel_values'][0].to(self.model.device,
                                                                                                       dtype=torch.float16)
                    special_token += [DEFAULT_IMAGE_TOKEN]
                elif os.path.splitext(file)[-1].lower() in video_ext:
                    file = self.video_processor(file, return_tensors='pt')['pixel_values'][0].to(self.model.device,
                                                                                            dtype=torch.float16)
                    special_token += [DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames
                else:
                    raise ValueError(
                        f'Support video of {video_ext} and image of {image_ext}, but found {os.path.splitext(file)[-1].lower()}')
                print(file.shape)
                tensor.append(file)

            inp = question

            if file is not None:
                # first message
                if getattr(self.model.config, "mm_use_im_start_end", False):
                    inp = ''.join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN for i in special_token]) + '\n' + inp
                else:
                    inp = ''.join(special_token) + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                file = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=tensor,  # video as fake images
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")
            conv.messages[-1][-1] = output

            print("\n", {"prompt": prompt, "outputs": output}, "\n")
            outputs.append(output)
        return outputs
    