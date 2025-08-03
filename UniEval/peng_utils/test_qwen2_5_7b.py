import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from decord import VideoReader, cpu

import os
from peng_utils import get_image, contact_img

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class Testqwen2_5_7b:
    def __init__(self, device=None):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/Qwen2.5-VL-7B-Instruct")
        # min_pixels = 256 * 28 * 28
        # max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/Qwen2.5-VL-7B-Instruct") #,min_pixels=min_pixels, max_pixels=max_pixels)
        
        # 初始化设备信息
        self.device = 'cpu'
        self.dtype = torch.float32

    def move_to_device(self, device=None):
        if device is not None and hasattr(device, 'type') and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
            self.model = self.model.to(device)
            print(f"Model moved to GPU: {device}")
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model = self.model.to('cpu')
            print(f"Model moved to CPU")

    def generate_image(self, image, question):
        print("generate begins")
        print(f"image:{type(image)}\nquestion:{type(question)}{question}")
        outputs = []
        # image = contact_img(image)
        image_list_list = image
        question_list = question
        for image_list, question in zip(image_list_list, question_list):
            # img = get_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                }
            ]
            for img in image_list:
                messages[0]["content"].append({"type": "image","image": get_image(np.array(img)),"resized_height": 224,"resized_width": 224})
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # 将输入移动到与模型相同的设备
            inputs = inputs.to(self.device)

            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            outputs.append(output)

        print("generate finish")
        return outputs

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

