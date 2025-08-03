import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from decord import VideoReader, cpu

import os
from peng_utils import get_image, contact_img

from peft import PeftModel
def encode_video(video_path, MAX_NUM_FRAMES=64):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

class TestTunedMiniCPM_V_2_6:
    def __init__(self, device=None):
        model_path = "/home/fx/Exp2/video_model/MiniCPM-V/MODELS/"
        # path_to_adapter = "/home/fx/Exp2/video_model/MiniCPM-V/finetune/output1/output__lora/checkpoint-2000/"
        # path_to_adapter = "/home/fx/Exp2/video_model/MiniCPM-V/finetune/output2/output__lora/checkpoint-1600/"
        # path_to_adapter = "/home/fx/Exp2/video_model/MiniCPM-V/finetune/output3/output__lora/checkpoint-6000/"
        path_to_adapter = "/home/fx/Exp2/video_model/MiniCPM-V/finetune/output/output__lora/checkpoint-2000/"
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        print("model1")
        self.model = PeftModel.from_pretrained(
                            model,
                            path_to_adapter,
                            device_map="auto",
                            trust_remote_code=True
                        ).eval().cuda(device)
        print("model2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)






        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa',
                                          torch_dtype=torch.bfloat16)

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
        # print(image_list_list)
        print(question_list)
        print(np.array(image_list_list).shape)
        images = []
        # print(image_list)
        for image_list in image_list_list:
            tem_list = []
            for image in image_list:
                img = get_image(np.array(image))
                tem_list.append(img)
            images.append(tem_list)

        answers = []
        for image_list, question in zip(images, question_list):
            msgs = [{"role": "user", "content": image_list + [question]}]
            answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
            answers.append(answer)
        print("generate finish")
        return answers

    @torch.no_grad()
    def generate_clip(self, file_list, question_list):
        print("generate begins")
        answers = []
        for video_path, question in zip(file_list, question_list):
            frames = encode_video(video_path, MAX_NUM_FRAMES=15)
            msgs = [{"role": "user", "content": frames + [question]}]

            params = {}
            params["use_image_id"] = False
            params["max_slice_nums"] = 2  # use 1 if cuda OOM and video resolution > 448*448

            answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, **params)
            answers.append(answer)
        print("generate finish")
        return answers

