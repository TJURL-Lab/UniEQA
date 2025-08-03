import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
from decord import VideoReader, cpu
import math

import os
from peng_utils import get_image, contact_img

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class Testvebrain:
    def __init__(self, device=None):
        print("🚀 Loading vebrain with optimizations...")
        
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
                "/home/fx/Exp2/test/EmbodiedEval/msjeval/VeBrain",
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
                "/home/fx/Exp2/test/EmbodiedEval/msjeval/VeBrain",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            self.is_multi_gpu = False
        
        # 处理器
        self.processor = AutoProcessor.from_pretrained(
            "/home/fx/Exp2/test/EmbodiedEval/msjeval/VeBrain"
        )
        
        # 优化参数
        self.batch_size = 1  # 单样本模式，最佳稳定性
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

    def prepare_batch_inputs(self, image_list_list, question_list):
        """批量准备输入数据"""
        batch_messages = []
        batch_texts = []
        batch_image_inputs = []
        batch_video_inputs = []
        
        for image_list, question in zip(image_list_list, question_list):
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            # 添加图像，限制尺寸以提高效率
            for img in image_list:
                messages[0]["content"].append({
                    "type": "image",
                    "image": get_image(np.array(img)),
                    "resized_height": 224,
                    "resized_width": 224
                })
            
            # 生成文本模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 调试输出 - 检查prompt内容（可选）
            # print(f"🔍 Debug prompt preview: {text[-200:] if len(text) > 200 else text}")
            
            # 处理视觉信息 - 添加错误处理
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                # 确保返回值不为None
                image_inputs = image_inputs if image_inputs is not None else []
                video_inputs = video_inputs if video_inputs is not None else []
            except Exception as e:
                print(f"⚠️ Warning: process_vision_info failed: {e}")
                image_inputs, video_inputs = [], []
            
            batch_messages.append(messages)
            batch_texts.append(text)
            batch_image_inputs.extend(image_inputs)
            batch_video_inputs.extend(video_inputs)
        
        return batch_texts, batch_image_inputs, batch_video_inputs

    @torch.no_grad()  # 添加无梯度计算，节省内存
    def generate_image_batch(self, image_list_list, question_list):
        """批量推理优化版本"""
        print(f"🔄 Batch inference: {len(image_list_list)} samples")
        
        all_outputs = []
        
        # 分批处理以避免内存溢出
        for i in range(0, len(image_list_list), self.batch_size):
            batch_images = image_list_list[i:i + self.batch_size]
            batch_questions = question_list[i:i + self.batch_size]
            
            print(f"  Processing batch {i//self.batch_size + 1}/{math.ceil(len(image_list_list)/self.batch_size)}")
            
            try:
                # 准备批量输入
                batch_texts, batch_image_inputs, batch_video_inputs = self.prepare_batch_inputs(
                    batch_images, batch_questions
                )
                
                # 处理器编码 - 修复空列表问题
                inputs = self.processor(
                    text=batch_texts,
                    images=batch_image_inputs if batch_image_inputs else None,
                    videos=None,  # 暂时禁用videos避免空列表错误
                    padding=True,
                    return_tensors="pt",
                )
                
                # 将输入移动到正确设备 - 修复设备警告
                inputs = inputs.to(self.device if not self.is_multi_gpu else 'cuda:0')
                
                # 清理模型状态 - 重要！防止批次间污染
                if hasattr(self.model, 'past_key_values'):
                    self.model.past_key_values = None
                if hasattr(self.model, '_past_key_values'):
                    self.model._past_key_values = None
                
                # 强制清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 批量推理 - 优化性能参数
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # 确定性推理，更快
                    num_beams=1,      # 不使用beam search，更快
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # 降低重复惩罚，加速
                    # no_repeat_ngram_size=3,  # 暂时禁用，提升速度
                    use_cache=True,  # 重新启用缓存，加速推理
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
                
                # 调试和清理输出
                cleaned_outputs = []
                for idx, output in enumerate(batch_outputs):
                    print(f"🔍 Raw output {idx}: '{output[:50]}...' (len: {len(output)})")
                    
                    # 清理输出
                    output = output.strip()
                    
                    # 检查并清理异常的addCriterion前缀
                    if output.startswith('addCriterion'):
                        print(f"⚠️  Sample {idx}: Detected addCriterion prefix, removing...")
                        # 找到第一个换行符或者合理的分割点
                        lines = output.split('\n')
                        if len(lines) > 1:
                            # 跳过第一行（addCriterion行）
                            output = '\n'.join(lines[1:]).strip()
                        else:
                            # 如果没有换行符，尝试找到addCriterion后面的内容
                            if len(output) > 12:  # "addCriterion" is 12 chars
                                output = output[12:].strip()
                        
                        print(f"✅ Sample {idx}: Cleaned to: '{output[:50]}...'")
                    
                    if not output:
                        output = "Unable to process this question."
                    
                    cleaned_outputs.append(output)
                
                all_outputs.extend(cleaned_outputs)
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error in batch {i//self.batch_size + 1}: {e}")
                # 降级到单个样本处理
                for j, (img_list, question) in enumerate(zip(batch_images, batch_questions)):
                    try:
                        output = self.generate_image_single(img_list, question)
                        all_outputs.append(output)
                    except Exception as e2:
                        print(f"❌ Error in single sample {i+j}: {e2}")
                        all_outputs.append(f"Error: {str(e2)}")
        
        print(f"✅ Batch inference completed: {len(all_outputs)} outputs")
        return all_outputs

    def generate_image_single(self, image_list, question):
        """单个样本推理（降级方案）"""
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
        
        # 调试输出 - 检查单个样本prompt内容（可选）
        # print(f"🔍 Single debug prompt preview: {text[-200:] if len(text) > 200 else text}")
        
        try:
            image_inputs, video_inputs = process_vision_info(messages)
            image_inputs = image_inputs if image_inputs is not None else []
            video_inputs = video_inputs if video_inputs is not None else []
        except Exception as e:
            print(f"⚠️ Warning: process_vision_info failed: {e}")
            image_inputs, video_inputs = [], []
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=None,  # 暂时禁用videos避免空列表错误
            padding=True,
            return_tensors="pt",
        )
        
        # 将输入移动到正确设备 - 修复设备警告
        inputs = inputs.to(self.device if not self.is_multi_gpu else 'cuda:0')
        
        # 清理模型状态 - 单个样本也需要
        if hasattr(self.model, 'past_key_values'):
            self.model.past_key_values = None
        if hasattr(self.model, '_past_key_values'):
            self.model._past_key_values = None
        
        # 强制清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # 确定性推理
            num_beams=1,      # 不使用beam search
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            repetition_penalty=1.2,  # 降低重复惩罚，加速
            # no_repeat_ngram_size=3,  # 暂时禁用，提升速度
            use_cache=True,  # 重新启用缓存，加速推理
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 调试和清理单个样本输出（静默模式）
        # print(f"🔍 Single output raw: '{output[:50]}...' (len: {len(output)})")
        
        output = output.strip()
        
        # 静默检查并清理异常的addCriterion前缀
        if output.startswith('addCriterion'):
            # print("⚠️  Single sample: Detected addCriterion prefix, removing...")
            lines = output.split('\n')
            if len(lines) > 1:
                output = '\n'.join(lines[1:]).strip()
            else:
                if len(output) > 12:
                    output = output[12:].strip()
            # print(f"✅ Single sample: Cleaned to: '{output[:50]}...'")
        
        if not output:
            output = "Unable to process this question."
        
        return output

    def generate_image(self, image_list_list, question_list):
        """主要接口：自动选择批量或单个推理"""
        print("🚀 vebrain inference begins")
        print(f"📊 Input: {len(image_list_list)} samples")
        
        if len(image_list_list) > 1:
            # 使用批量推理
            outputs = self.generate_image_batch(image_list_list, question_list)
        else:
            # 单个样本
            outputs = [self.generate_image_single(image_list_list[0], question_list[0])]
        
        print("✅ vebrain inference finished")
        return outputs 