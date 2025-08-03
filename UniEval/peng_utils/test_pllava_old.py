import sys

from peng_utils import contact_img, get_image

sys.path.append("./peng_utils")
from argparse import ArgumentParser
import copy
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from pllava.utils.easydict import EasyDict
from pllava.tasks.eval.model_utils import load_pllava
from pllava.tasks.eval.eval_utils import (
    ChatPllava,
    conv_plain_v1,
    Conversation,
    conv_templates
)
from pllava.tasks.eval.demo import pllava_theme

import torch

SYSTEM="""You are a powerful Video Magic ChatBot, a large vision-language assistant. 
You are able to understand the video content that the user provides and assist the user in a video-language related task.
The user might provide you with the video and maybe some extra noisy information to help you out or ask you a question. Make use of the information in a proper way to be competent for the job.
### INSTRUCTIONS:
1. Follow the user's instruction.
2. Be critical yet believe in yourself.
"""
INIT_CONVERSATION: Conversation = conv_plain_v1.copy()

class TestPLLaVA:
    def __init__(self, device=None) -> None:
        model_dir = "/home/fx/Exp2/test/EmbodiedEval/MODELS/pllava-7b"
        num_frames = 16
        use_lora = True
        weight_dir = "/home/fx/Exp2/test/EmbodiedEval/MODELS/pllava-7b"
        lora_alpha = 4
        use_multi_gpus = False
        self.conv_mode = "plain"
        self.device = device

        self.model, self.processor = load_pllava(
            model_dir, num_frames,
            use_lora=use_lora,
            weight_dir=weight_dir,
            lora_alpha=lora_alpha,
            use_multi_gpus=use_multi_gpus)

        if not use_multi_gpus:
            self.model = self.model.to(device)
        self.chat = ChatPllava(self.model, self.processor)


    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            device = device
        else:
            self.dtype = torch.float16
            device = 'cpu'
        self.model = self.model.to(device, dtype=self.dtype)


    def __device__(self):
        return self.device


    @torch.no_grad()
    def generate_image(self, image_list_list, question_list, num_segments=None, img_list=None, num_beams=1, temperature=1.0):
        outputs = []
        image_list = contact_img(image_list_list)
        # print(f"generate_image image_list:{image_list}")
        for image, user_message in zip(image_list, question_list):
            INIT_CONVERSATION = conv_templates[self.conv_mode]
            chat_state = INIT_CONVERSATION.copy()
            img_list = []
            image = get_image(image)
            # upload img TODO: handle image_list
            llm_message, img_list, chat_state = self.chat.upload_img(image, chat_state, img_list)
            chat_state = self.chat.ask(user_message, chat_state, SYSTEM)
            # answer
            llm_message, llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list,
                                                                          max_new_tokens=200, num_beams=num_beams,
                                                                          temperature=temperature, media_type='image')
            llm_message = llm_message.replace("<s>", "")  # handle <s>

            outputs.append(llm_message)

        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")

        return outputs


    @torch.no_grad()
    def generate_clip(self, files_list, question_list, num_segments=None, img_list=None, num_beams=1, temperature=1.0):
        outputs = []
        for file, user_message in zip(files_list, question_list):
            INIT_CONVERSATION = conv_templates[self.conv_mode]
            chat_state = INIT_CONVERSATION.copy()
            img_list = []
            # upload img
            llm_message, img_list, chat_state = self.chat.upload_video(file, chat_state, img_list, num_segments)
            chat_state = self.chat.ask(user_message, chat_state, SYSTEM)
            # answer
            llm_message, llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list,
                                                                          max_new_tokens=200, num_beams=num_beams,
                                                                          temperature=temperature, media_type='video')
            llm_message = llm_message.replace("<s>", "")  # handle <s>

            outputs.append(llm_message)

        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")

        return outputs