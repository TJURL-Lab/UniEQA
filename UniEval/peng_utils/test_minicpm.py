from io import BytesIO

from .minicpm import OmniLMMChat, img2base64
import torch
import json
import base64
import os

from peng_utils import get_image, contact_img


class TestMiniCPM:
    def __init__(self, device=None):
        model_path = 'openbmb/MiniCPM-Llama3-V-2_5'
        self.chat_model = OmniLMMChat(model_path, device)


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
        images = []
        # print(image_list)
        for image in image_list:
            img = get_image(image)
            # print(img)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")  # 使用你图像的实际格式（JPEG, PNG, 等）
            img = base64.b64encode(buffered.getvalue())
            images.append(img)

        answers = []
        for image, question in zip(images, question_list):

            msgs = [{"role": "user", "content": question}]
            inputs = {"image": image, "question": json.dumps(msgs)}
            answer = self.chat_model.chat(inputs)
            answers.append(answer)
        print("generate finish")
        return answers

