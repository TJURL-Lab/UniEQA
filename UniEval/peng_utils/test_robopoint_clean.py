"""
RoboPoint测试类 - 使用官方实现，支持多GPU
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import threading

class TestRoboPoint:
    def __init__(self, device=None):
        """使用官方方式加载RoboPoint模型"""
        # 简化device处理，直接传给官方builder
        if isinstance(device, torch.device):
            device_map = f"cuda:{device.index}" if device.index is not None else "cuda:0"
            self.device = device
        elif isinstance(device, str):
            device_map = device if device in ["auto", "cpu"] else device
            self.device = torch.device(device_map if device_map != "auto" else "cuda:0")
        elif isinstance(device, int):
            device_map = f"cuda:{device}"
            self.device = torch.device(device_map)
        elif device is None:
            device_map = "auto"  # 让transformers自动分配多GPU
            self.device = torch.device("cuda:0")
        else:
            device_map = "auto"
            self.device = torch.device("cuda:0")
        
        self.model_path = "/home/fx/Exp2/video_model/RoboPoint-8B"
        self.model_name = "RoboPoint-8B"
        self.device_map = device_map
        
        print(f"🚀 初始化TestRoboPoint，device_map: {device_map}")
        self._load_model()
        
    def _load_model(self):
        """使用官方方式加载模型"""
        try:
            # 导入官方builder
            sys.path.append('/home/fx/Exp2/video_model/RoboPoint')
            from robopoint.model.builder import load_pretrained_model
            
            # 使用官方的load_pretrained_model函数
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=self.model_name,
                load_8bit=False,
                load_4bit=False,
                device_map=self.device_map,  # "auto" 或具体设备
                device="cuda",
                use_flash_attn=False
            )
            
            self.context_len = context_len
            print("✅ TestRoboPoint模型加载成功")
            
            # 打印设备映射信息
            if hasattr(self.model, 'hf_device_map'):
                print("📍 模型设备映射:")
                device_info = {}
                for layer, device in self.model.hf_device_map.items():
                    if device not in device_info:
                        device_info[device] = []
                    device_info[device].append(layer)
                
                for device, layers in device_info.items():
                    print(f"   GPU {device}: {len(layers)} 层")
                    
        except Exception as e:
            print(f"❌ TestRoboPoint模型加载失败: {e}")
            raise e
    
    def generate_image(self, image_list_list, question_list):
        """生成图像响应 - 使用官方方式"""
        if not isinstance(image_list_list, list):
            image_list_list = [image_list_list]
        if not isinstance(question_list, list):
            question_list = [question_list]
            
        results = []
        
        for i, (images, question) in enumerate(zip(image_list_list, question_list)):
            try:
                # 使用官方的处理方式
                result = self._generate_with_official_method(images, question)
                results.append(result)
                print(f"✅ 第{i+1}个样本生成成功")
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                results.append(error_msg)
                print(f"❌ 第{i+1}个样本生成失败: {e}")
        
        return results
    
    def _generate_with_official_method(self, images, question):
        """使用官方方法生成响应"""
        from robopoint.mm_utils import process_images, tokenizer_image_token
        from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        
        # 处理图像
        if images is not None and len(images) > 0:
            # 转换为PIL图像列表
            pil_images = []
            if isinstance(images, list):
                for img in images:
                    pil_img = self._convert_to_pil(img)
                    if pil_img:
                        pil_images.append(pil_img)
            else:
                pil_img = self._convert_to_pil(images)
                if pil_img:
                    pil_images = [pil_img]
            
            if not pil_images:
                return "Error: No valid images provided"
            
            # 使用官方的process_images
            processed_images = process_images(pil_images, self.image_processor, self.model.config)
            
            # 将图像移动到正确设备
            if isinstance(processed_images, list):
                processed_images = [img.to(self.model.device, dtype=torch.float16) for img in processed_images]
            else:
                processed_images = processed_images.to(self.model.device, dtype=torch.float16)
            
            # 构建带图像token的prompt
            image_tokens = DEFAULT_IMAGE_TOKEN * len(pil_images)
            prompt = f"{image_tokens}\n{question}"
            
        else:
            processed_images = None
            prompt = question
        
        # 使用官方的tokenizer_image_token
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)
        
        # 生成响应
        with torch.inference_mode():
            if processed_images is not None:
                output_ids = self.model.generate(
                    input_ids,
                    images=processed_images,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=512,
                    use_cache=True
                )
            else:
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=512,
                    use_cache=True
                )
        
        # 解码响应
        response = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def _convert_to_pil(self, image):
        """转换图像为PIL格式"""
        try:
            if isinstance(image, str):
                # 文件路径
                if os.path.exists(image):
                    return Image.open(image).convert('RGB')
                else:
                    print(f"⚠️ 图像文件不存在: {image}")
                    return None
            elif isinstance(image, Image.Image):
                return image.convert('RGB')
            elif isinstance(image, np.ndarray):
                return Image.fromarray(image).convert('RGB')
            elif isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.dim() == 3 and image.shape[0] in [1, 3]:
                    image = image.permute(1, 2, 0)
                image_np = image.cpu().numpy()
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).astype(np.uint8)
                return Image.fromarray(image_np).convert('RGB')
            else:
                print(f"⚠️ 不支持的图像类型: {type(image)}")
                return None
        except Exception as e:
            print(f"⚠️ 图像转换失败: {e}")
            return None
    
    def clear_cache(self):
        """清理缓存"""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        print("🧹 GPU缓存已清理")

if __name__ == "__main__":
    # 测试代码
    print("🚀 开始测试RoboPoint")
    
    # 初始化模型
    model = TestRoboPoint(device="auto")
    
    # 测试纯文本生成
    result = model.generate_image([], ["Hello, how are you?"])
    print(f"📝 纯文本结果: {result}")
    
    print("✅ 测试完成") 