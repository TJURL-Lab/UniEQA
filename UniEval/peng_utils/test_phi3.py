import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from decord import VideoReader, cpu

import os
from peng_utils import get_image, contact_img

from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

class TestPhi3:
    def __init__(self, device=None):
        model_id = "microsoft/Phi-3-vision-128k-instruct"
        torch.manual_seed(0)
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
    
    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):
        print("generate begins")
        image_list = contact_img(image_list_list)

        answers = []
        for image, question in zip(image_list, question_list):
            messages = [
                {"role": "user", "content": "<|image_1|>\n"+question}
            ]
            image = get_image(image)
            prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device)
            generation_args = {
                "max_new_tokens": 500,
                "temperature": 0.0,
                "do_sample": False,
            }
            generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            answer = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answers.append(answer)
        print("generate finish")
        return answers

    # @torch.no_grad()
    # def generate_clip(self, file_list, question_list):
    #     print("generate begins")
    #     answers = []
    #     for video_path, question in zip(file_list, question_list):
    #         frames = encode_video(video_path, MAX_NUM_FRAMES=15)
    #         msgs = [{"role": "user", "content": frames + [question]}]
    #
    #         params = {}
    #         params["use_image_id"] = False
    #         params["max_slice_nums"] = 2  # use 1 if cuda OOM and video resolution > 448*448
    #
    #         answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, **params)
    #         answers.append(answer)
    #     print("generate finish")
    #     return answers

