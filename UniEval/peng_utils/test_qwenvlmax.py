import base64
import json
import os
import sys
import time
import traceback
from io import BytesIO

import requests
import asyncio
from openai import AsyncOpenAI

sys.path.append("./peng_utils")
import numpy as np
import torch


from . import get_image

# CFG_PATH = 'models/minigpt4/minigpt4_eval.yaml'


def read_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
    return config


cfg = read_config("./peng_utils/openai_cfg.json")["apis"][3]
API_KEY = cfg["api_key"]
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["http_proxy"] = cfg["http_proxy"]
os.environ["https_proxy"] = cfg["https_proxy"]
### TODO #open source: remove my Proxy
Proxy = cfg["proxy"]


class TestQwenVLMax :
    def __init__(self, device=None):
        self.model = "qwen-vl-max"    # "gpt-4-vision-preview"
        self.api_key = API_KEY
        self.proxy = Proxy  # 统一去掉末尾斜杠

        # -------- OpenAI Async Client --------
        # base_url 需指向 /v1 结尾
        base_url = self.proxy

        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=500,
        )

        # 并发上限 5
        self.sema = asyncio.Semaphore(5)

        self.device = device


    def move_to_device(self, device):
        return


    def _images_to_base64(self, image_list):
        """把图像列表转 base64 字符串列表"""
        encoded = []
        for image in image_list:
            img = get_image(np.array(image))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return encoded

    async def answer_async(self, base64_image_list, context, max_tokens=300):
        # 构建消息
        content = [{"type": "text", "text": context}]
        for b64 in base64_image_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        # 自动补全 URL
        # AsyncOpenAI 已设置 base_url，因此无需关心具体路径
        for attempt in range(10):
            try:
                async with self.sema:
                    resp = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=max_tokens,
                    )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"[async] retry {attempt}: {e}")
                await asyncio.sleep(0.5)
        return "请求失败"

    async def _batch_async(self, image_list_list, question_list):
        tasks = []
        for imgs, q in zip(image_list_list, question_list):
            tasks.append(self.answer_async(imgs, q))
        return await asyncio.gather(*tasks)

    def generate_image(self, image_list_list, question_list):
        print("generate begins (async batch)")

        # 先把所有图像转换为 base64
        b64_batches = [self._images_to_base64(imgs) for imgs in image_list_list]

        answers = list(asyncio.run(self._batch_async(b64_batches, question_list)))

        print("generate finish")
        return answers

    # 兼容旧接口：如果外部直接调用 answer，同步封装异步
    def answer(self, base64_image_list, context, max_tokens=300):
        return asyncio.run(self.answer_async(base64_image_list, context, max_tokens))
