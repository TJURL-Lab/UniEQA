import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from decord import VideoReader, cpu
import math
import traceback

import os
from peng_utils import get_image, contact_img

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class Testqwen2_5_7b_fixed:
    def __init__(self, device=None):
        print("🚀 Loading Qwen2.5-VL-7B-Instruct with optimizations and fixes...")
        
        # 确定设备
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                self.dtype = torch.float16
            else:
                device = torch.device('cpu')
                self.dtype = torch.float32
        else:
            self.dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        
        self.device = device
        print(f"🎯 Target device: {device}, dtype: {self.dtype}")
        
        # 使用优化的设备映射直接加载到GPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # 多GPU情况：使用自动设备映射
            print(f"🔥 Multi-GPU setup detected: {torch.cuda.device_count()} GPUs")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/Qwen2.5-VL-7B-Instruct",
                torch_dtype=self.dtype,
                device_map='auto',  # 自动分布到多GPU
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.is_multi_gpu = True
        else:
            # 单GPU情况：直接加载到指定设备
            print(f"📱 Single GPU setup")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/Qwen2.5-VL-7B-Instruct",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            self.is_multi_gpu = False
        
        # 处理器
        self.processor = AutoProcessor.from_pretrained(
            "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/Qwen2.5-VL-7B-Instruct"
        )
        
        # 优化参数
        self.batch_size = 4  # 降低批量大小避免OOM
        self.max_new_tokens = 128
        
        print(f"✅ Model loaded successfully! Multi-GPU: {self.is_multi_gpu}")

    def move_to_device(self, device=None):
        """保持兼容性，但已经在初始化时优化了设备加载"""
        if device is not None and not self.is_multi_gpu:
            if hasattr(device, 'type') and 'cuda' in device.type:
                self.dtype = torch.float16
                self.device = device
                self.model = self.model.to(device)
                print(f"Model moved to GPU: {device}")
            else:
                self.dtype = torch.float32
                self.device = 'cpu'
                self.model = self.model.to('cpu')
                print(f"Model moved to CPU")
        elif self.is_multi_gpu:
            print("Multi-GPU model already optimally distributed")

    def prepare_single_input(self, image_list, question):
        """准备单个样本的输入 - 修复版本"""
        try:
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            # 添加图像
            for img in image_list:
                if img is not None:
                    messages[0]["content"].append({
                        "type": "image",
                        "image": get_image(np.array(img)),
                        "resized_height": 224,
                        "resized_width": 224
                    })
            
            # 生成文本模板 - 添加英文提示
                    text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 添加英文引导
        if not text.endswith("Assistant: "):
            text = text.rstrip() + " Please respond in English: "
            
            # 处理视觉信息 - 添加错误处理
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                # 确保返回值不为None
                image_inputs = image_inputs if image_inputs is not None else []
                video_inputs = video_inputs if video_inputs is not None else []
            except Exception as e:
                print(f"⚠️ Warning: process_vision_info failed for sample: {e}")
                image_inputs, video_inputs = [], []
            
            return text, image_inputs, video_inputs, messages
            
        except Exception as e:
            print(f"❌ Error preparing input: {e}")
            return None, [], [], None

    @torch.no_grad()  # 添加无梯度计算，节省内存
    def generate_image_batch_fixed(self, image_list_list, question_list):
        """修复的批量推理版本"""
        print(f"🔄 Fixed batch inference: {len(image_list_list)} samples")
        
        all_outputs = []
        
        # 分批处理以避免内存溢出
        for i in range(0, len(image_list_list), self.batch_size):
            batch_images = image_list_list[i:i + self.batch_size]
            batch_questions = question_list[i:i + self.batch_size]
            
            print(f"  Processing batch {i//self.batch_size + 1}/{math.ceil(len(image_list_list)/self.batch_size)}")
            
            try:
                # 分别准备每个样本的输入
                batch_texts = []
                all_image_inputs = []
                all_video_inputs = []
                valid_samples = []
                
                for j, (img_list, question) in enumerate(zip(batch_images, batch_questions)):
                    text, image_inputs, video_inputs, messages = self.prepare_single_input(img_list, question)
                    if text is not None:
                        batch_texts.append(text)
                        all_image_inputs.extend(image_inputs)
                        all_video_inputs.extend(video_inputs)
                        valid_samples.append(j)
                
                if not batch_texts:
                    print("❌ No valid samples in this batch, skipping...")
                    continue
                
                # 处理器编码 - 添加错误处理
                try:
                    inputs = self.processor(
                        text=batch_texts,
                        images=all_image_inputs if all_image_inputs else None,
                        videos=all_video_inputs if all_video_inputs else None,
                        padding=True,
                        return_tensors="pt",
                    )
                except Exception as e:
                    print(f"❌ Processor encoding failed: {e}")
                    # 降级到单样本处理
                    raise e
                
                # 将输入移动到正确设备
                if not self.is_multi_gpu:
                    inputs = inputs.to(self.device)
                
                # 批量推理 - 优化参数
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # 确定性推理，更快
                    num_beams=1,      # 不使用beam search，更快
                    use_cache=True,   # 使用缓存加速
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # 增加重复惩罚
                    length_penalty=1.0,      # 长度惩罚
                    # 移除temperature避免警告
                )
                
                # 解码输出
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                batch_outputs = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                # 确保输出数量匹配
                for j in range(len(batch_images)):
                    if j in valid_samples:
                        idx = valid_samples.index(j)
                        all_outputs.append(batch_outputs[idx] if idx < len(batch_outputs) else "Error: No output")
                    else:
                        all_outputs.append("Error: Invalid input")
                
                print(f"✅ Batch {i//self.batch_size + 1} completed successfully")
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error in batch {i//self.batch_size + 1}: {e}")
                print(f"🔄 Falling back to single sample processing...")
                
                # 降级到单个样本处理
                for j, (img_list, question) in enumerate(zip(batch_images, batch_questions)):
                    try:
                        output = self.generate_image_single_fast(img_list, question)
                        all_outputs.append(output)
                    except Exception as e2:
                        print(f"❌ Error in single sample {i+j}: {e2}")
                        all_outputs.append(f"Error: {str(e2)}")
        
        print(f"✅ Fixed batch inference completed: {len(all_outputs)} outputs")
        return all_outputs

    @torch.no_grad()
    def generate_image_single_fast(self, image_list, question):
        """优化的单样本推理"""
        try:
            text, image_inputs, video_inputs, messages = self.prepare_single_input(image_list, question)
            
            if text is None:
                return "Error: Failed to prepare input"
            
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            )
            
            if not self.is_multi_gpu:
                inputs = inputs.to(self.device)
            
            # 优化的生成参数
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 添加重复惩罚
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output
            
        except Exception as e:
            print(f"❌ Single sample error: {e}")
            return f"Error: {str(e)}"

    def generate_image(self, image_list_list, question_list):
        """主要接口：优化版本"""
        print("🚀 Qwen2.5-VL fixed and optimized inference begins")
        print(f"📊 Input: {len(image_list_list)} samples")
        
        try:
            if len(image_list_list) > 1:
                # 使用修复的批量推理
                outputs = self.generate_image_batch_fixed(image_list_list, question_list)
            else:
                # 单个样本
                outputs = [self.generate_image_single_fast(image_list_list[0], question_list[0])]
            
            print("✅ Qwen2.5-VL fixed and optimized inference finished")
            return outputs
            
        except Exception as e:
            print(f"❌ Critical error in generate_image: {e}")
            traceback.print_exc()
            # 最后的降级方案
            return [f"Critical Error: {str(e)}" for _ in range(len(image_list_list))] 