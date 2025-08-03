import inspect
import sys

sys.path.insert(0, './peng_utils')
import os
import torch
import pickle
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import sys
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torch
from lavis.models import load_model_and_preprocess
from concurrent.futures import ThreadPoolExecutor

from peng_utils import get_image


class TestInstructBLIP:
    def __init__(self, device=None) -> None:
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device
        )
        # print(f"testblip MODEL:{self.model}\ntestblip VIS:{self.vis_processors}")
        # print(f"GENERATE SIGNATURE: {inspect.signature(self.model.generate)}")  # 打印函数签名
        # print(f"GENERATE getsource: {inspect.getsource(self.model.generate)}")
        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float32
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        # self.model = self.model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):

        images_list = []
        for image_list in image_list_list:
            images = []
            for image in image_list:
                img = get_image(np.array(image))
                img = self.vis_processors["eval"](img)
                images.append(img)
            images = torch.stack(images, 1)
            images_list.append(images)
        images_list = torch.from_numpy(np.array(images_list)).to(self.device)
        pred_responses = self.model.generate({"image": images_list, "prompt": question_list})


        return pred_responses


