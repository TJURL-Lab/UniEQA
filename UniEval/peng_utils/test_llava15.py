import sys
sys.path.append("./peng_utils/")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peng_utils import contact_img, get_image

import torch
# import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor

from llava15.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava15.conversation import conv_templates, SeparatorStyle
from llava15.model.builder import load_pretrained_model
from llava15.utils import disable_torch_init
from llava15.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


class TestLLaVA15:
    def __init__(self, device=None):
        model_path = "/home/sdd/fx/SpaceMantis/"
        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)
        self.model.eval()

        if device is not None:
            self.model.to(device)
            self.move_to_device(device)

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):
        print("generate begins")
        pred_responses = self.batch_generate(self.model, self.image_processor, self.tokenizer, image_list_list, question_list)
        print("generate finish")
        return pred_responses


    def batch_generate(self, model, image_processor, tokenizer, batch_img_list, batch_context, max_new_tokens=128,
                       num_beams=1):
        device = self.device
        batch_img_embd_list = []
        print(f"batch_img_list_length:{len(batch_img_list)}")
        print(f"img_list:{len(batch_img_list[0])}")
        for j, img_list in enumerate(batch_img_list):
            for i, img in enumerate(img_list):
                batch_img_list[j][i] = get_image(np.array(img))
        print(batch_img_list)
        for img_list in batch_img_list:
            list_embd = image_processor.preprocess(img_list, return_tensors="pt")["pixel_values"]
            batch_img_embd_list.append(list_embd)
        batch_imbd = torch.stack(batch_img_embd_list, dim=0).to(device)
        print(f"batch_imbd.shape: {batch_imbd.shape}, batch_context: {batch_context}")
        input_ids = tokenizer(batch_context).input_ids
        # print(input_ids.shape)
        max_prompt_size = max([len(input_id) for input_id in input_ids])
        for i in range(len(input_ids)):
            padding_size = max_prompt_size - len(input_ids[i])
            input_ids[i] = [tokenizer.pad_token_id] * padding_size + input_ids[i]
        input_ids = torch.as_tensor(input_ids).to(device)
        print(f"input_ids.shape:{input_ids.shape}")
        stop_str = "###"
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=batch_imbd.half().cuda(device),
                do_sample=False,  ### True
                max_new_tokens=max_new_tokens,
                use_cache=True)
            len_input = torch.as_tensor(input_ids).to(device).size()[1]
            print(f"len_input:{len_input}\noutput_ids:{output_ids.shape}")
            output_ids = output_ids[:, len_input:]
            batch_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for i, outputs in enumerate(batch_outputs):
                # outputs = outputs.split('###Assistant:')[-1]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                batch_outputs[i] = outputs
                print(f"{i}th output:{outputs}")
        return batch_outputs