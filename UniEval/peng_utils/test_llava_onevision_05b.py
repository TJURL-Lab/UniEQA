import os
import sys
import warnings
import copy
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import requests
from io import BytesIO

warnings.filterwarnings("ignore")

from peng_utils import contact_img, get_image

sys.path.append("./peng_utils")
sys.path.insert(0, "/home/fx/Exp2/video_model/LLaVA-NeXT")

# LLaVA imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_video(video_path, fps=1):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

class TestLLaVAOneVision:
    def __init__(self, device=None) -> None:
        disable_torch_init()
        
        # 基于官方教程的配置
        model_path = "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/llava-onevision-qwen2-0.5b-ov"
        model_name = "llava_qwen"  # 官方指定的模型名称
        device_map = "auto"
        
        # 显式禁用 FlashAttention2
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": None,  # 使用eager attention避免flash attention
            "torch_dtype": torch.float16,
        }
        
        self.device = device if device is not None else "cuda"
        
        print(f"🔄 开始加载LLaVA-OneVision模型: {model_path}")
        print(f"使用配置 - 模型名称: {model_name}, attention: None")
        
        try:
            # 先检查模型的实际名称
            detected_model_name = get_model_name_from_path(model_path)
            print(f"检测到的模型名称: {detected_model_name}")
            
            # 使用检测到的模型名称
            self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
                model_path, 
                None, 
                detected_model_name,  # 使用检测到的模型名称
                device_map=device_map, 
                **llava_model_args
            )
            
            self.model.eval()
            print("✅ 模型加载成功!")
            
        except Exception as e:
            print(f"❌ 第一次加载失败: {e}")
            # 尝试最简单的加载方式
            try:
                print("🔄 尝试最简单的加载方式...")
                # 使用基本的 llava 模型名称和简单参数
                simple_args = {
                    "multimodal": True,
                    "attn_implementation": None,  # 保持为None
                    "torch_dtype": torch.float16,
                }
                self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
                    model_path, 
                    None, 
                    "llava",  # 使用基本的llava名称
                    device_map=None,  # 不使用device_map
                    **simple_args
                )
                self.model.eval()
                print("✅ 简单方式加载成功!")
            except Exception as e2:
                print(f"❌ 所有加载方式都失败: {e2}")
                raise e2
        
        # 对话模板配置
        self.conv_template = "qwen_1_5"  # 官方指定的对话模板
        
        # 其他配置
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.top_p = 0.7

    def move_to_device(self, device):
        """兼容性方法，实际上模型已经通过device_map加载"""
        self.device = device
        pass

    def __device__(self):
        return self.device

    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):
        outputs = []
        
        # 使用和 test_llava.py 相同的方式预处理图像
        processed_images = contact_img(image_list_list)
        
        for image, question in zip(processed_images, question_list):
            try:
                # 转换图像格式 - 模仿 test_llava.py 的方式
                img = get_image(image)
                images = [img]  # LLaVA-OneVision 通常处理单个图像
                
                # 使用官方的process_images方法
                image_tensors = process_images(images, self.image_processor, self.model.config)
                image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
                
                # 准备对话，使用官方的方式
                conv = copy.deepcopy(conv_templates[self.conv_template])
                
                # 构建问题，根据图像数量调整
                if len(images) == 1:
                    question_with_image = DEFAULT_IMAGE_TOKEN + "\n" + question
                else:
                    # 多图像情况
                    image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(images))
                    question_with_image = image_tokens + "\n" + question
                
                conv.append_message(conv.roles[0], question_with_image)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                # Tokenize
                input_ids = tokenizer_image_token(
                    prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(self.device)
                
                # 图像尺寸
                image_sizes = [image.size for image in images]
                
                # 生成回答 - 使用官方的参数
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensors,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=self.max_new_tokens,
                    )
                
                # 解码输出
                text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                output = text_outputs[0].strip()
                
                # 清理输出（移除prompt部分）
                if prompt_question in output:
                    output = output.replace(prompt_question, "").strip()
                
                outputs.append(output)
                
                print(f"✅ 图像 {len(outputs)}: {question[:50]}... -> {output[:100]}...")
                
            except Exception as e:
                print(f"❌ 图像生成失败: {e}")
                outputs.append(f"Error processing image: {str(e)}")
        
        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")
        return outputs

    @torch.no_grad()
    def generate_clip(self, files_list, question_list):
        outputs = []
        
        for video_file, question in zip(files_list, question_list):
            try:
                # 加载视频帧或图像列表
                if isinstance(video_file, str):
                    # 如果是文件路径，加载视频
                    video_frames = load_video(video_file)
                    # 转换为PIL图像
                    images = [Image.fromarray(frame) for frame in video_frames]
                else:
                    # 如果已经是图像列表，使用和 test_llava.py 相同的处理方式
                    # 对于视频文件，也可能需要使用 contact_img 处理
                    processed_images = contact_img([video_file])  # 包装成需要的格式
                    img = get_image(processed_images[0])
                    images = [img]
                
                # 使用官方的process_images方法
                image_tensors = process_images(images, self.image_processor, self.model.config)
                image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
                
                # 准备对话
                conv = copy.deepcopy(conv_templates[self.conv_template])
                
                # 对于视频，使用单个图像token（LLaVA-OneVision支持视频理解）
                question_with_image = DEFAULT_IMAGE_TOKEN + "\n" + question
                
                conv.append_message(conv.roles[0], question_with_image)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                # Tokenize
                input_ids = tokenizer_image_token(
                    prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(self.device)
                
                # 图像尺寸
                image_sizes = [image.size for image in images]
                
                # 生成回答
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensors,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=self.max_new_tokens,
                    )
                
                # 解码输出
                text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                output = text_outputs[0].strip()
                
                # 清理输出（移除prompt部分）
                if prompt_question in output:
                    output = output.replace(prompt_question, "").strip()
                
                outputs.append(output)
                
                print(f"✅ 视频 {len(outputs)}: {question[:50]}... -> {output[:100]}...")
                
            except Exception as e:
                print(f"❌ 视频生成失败: {e}")
                outputs.append(f"Error processing video: {str(e)}")
        
        print("\n", {"prompt": question_list, "outputs": outputs}, "\n")
        return outputs 