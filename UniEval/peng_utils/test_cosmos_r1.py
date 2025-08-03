import numpy as np
from PIL import Image
from transformers import AutoTokenizer
import torch
from decord import VideoReader, cpu

import os
from peng_utils import get_image, contact_img

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class TestCosmosR1:
    def __init__(self, device=None):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/Cosmos-Reason1-7B", 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/Cosmos-Reason1-7B"
        )

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'

    def generate_image(self, image, question):
        print("generate begins")
        print(f"image:{type(image)}\nquestion:{type(question)}{question}")
        outputs = []
        image_list_list = image
        question_list = question
        
        for image_list, question in zip(image_list_list, question_list):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                }
            ]
            for img in image_list:
                messages[0]["content"].append({
                    "type": "image",
                    "image": get_image(np.array(img)),
                    "resized_height": 224,
                    "resized_width": 224
                })
            
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
            inputs = inputs.to("cuda")

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

    def generate_image_batch(self, image_list_list, question_list, batch_size=4):
        """
        批量并行推理，自动分批
        Args:
            image_list_list: List[List[image]]
            question_list: List[str]
            batch_size: int
        Returns:
            List[str] 推理结果
        """
        outputs = []
        total = len(image_list_list)
        for i in range(0, total, batch_size):
            batch_images = image_list_list[i:i+batch_size]
            batch_questions = question_list[i:i+batch_size]
            batch_outputs = self._generate_image_batch_core(batch_images, batch_questions)
            outputs.extend(batch_outputs)
        return outputs

    def _generate_image_batch_core(self, image_list_list, question_list):
        batch_texts = []
        batch_image_inputs = []
        batch_video_inputs = []
        
        for image_list, question in zip(image_list_list, question_list):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                }
            ]
            for img in image_list:
                messages[0]["content"].append({
                    "type": "image",
                    "image": get_image(np.array(img)),
                    "resized_height": 224,
                    "resized_width": 224
                })
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            batch_texts.append(text)
            batch_image_inputs.extend(image_inputs)
            batch_video_inputs.extend(video_inputs)
            
        inputs = self.processor(
            text=batch_texts,
            images=batch_image_inputs if batch_image_inputs else None,
            videos=batch_video_inputs if batch_video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
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

