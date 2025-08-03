import torch
import gc
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining
from transformers import LlavaOnevisionForConditionalGeneration
import os


class TestRoboBrain:
    """
    RoboBrain模型测试类
    基于BAAI/RoboBrain模型，支持图像和文本的多模态推理
    """
    
    def __init__(self, device=None, multi_gpu=None):
        """
        初始化RoboBrain模型
        
        Args:
            device: 计算设备，默认为cuda:0
            multi_gpu: 是否使用多卡，None表示自动检测
        """
        # 检测GPU设备
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 多卡设置
        if multi_gpu is None:
            self.use_multi_gpu = self.gpu_count > 1
        else:
            self.use_multi_gpu = multi_gpu and self.gpu_count > 1
            
        print(f"正在初始化RoboBrain模型，设备: {self.device}")
        print(f"检测到 {self.gpu_count} 个GPU设备")
        if self.use_multi_gpu:
            print(f"✅ 启用多卡推理，使用 {self.gpu_count} 个GPU")
        else:
            print(f"使用单卡推理")
        
        # 模型路径 - 优先使用本地模型
        local_model_path = "/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/RoboBrain"
        if os.path.exists(local_model_path):
            self.model_path = local_model_path
            print(f"使用本地模型: {self.model_path}")
        else:
            self.model_path = "BAAI/RoboBrain"
            print(f"使用HuggingFace模型: {self.model_path}")
        
        self.model = None
        self.processor = None
        self.is_model_parallel = False
        self.load_model()
        
    def load_model(self):
        """加载模型和处理器"""
        try:
            print("正在加载RoboBrain模型...")
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print("✅ 处理器加载成功")
            
            # 加载模型 - 根据多卡设置选择不同的加载方式
            if self.use_multi_gpu and self.gpu_count > 1:
                print(f"🔄 使用多卡加载模型到 {self.gpu_count} 个GPU...")
                # 使用device_map="auto"让transformers自动分配模型到多个GPU
                self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto"  # 自动分配到多个GPU
                )
                self.is_model_parallel = True
                print(f"✅ 模型已分布到多个GPU")
                
                # 打印设备分配信息
                if hasattr(self.model, 'hf_device_map'):
                    print("📊 设备分配映射:")
                    for layer, device in self.model.hf_device_map.items():
                        print(f"  {layer}: {device}")
                
            else:
                print(f"🔄 使用单卡加载模型到 {self.device}...")
                self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                # 移动到指定设备
                if torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                
                self.is_model_parallel = False
                print(f"✅ 模型已加载到 {self.device}")
                
            print("✅ RoboBrain模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: 图像，可以是PIL Image、numpy array、list或路径字符串
            
        Returns:
            PIL Image对象
        """
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image.astype('uint8')).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, list):
            # 如果是list，转换为numpy array然后处理
            image_array = np.array(image, dtype='uint8')
            return Image.fromarray(image_array).convert('RGB')
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
    
    def generate_image(self, image_list_list, question_list, max_new_tokens=512, temperature=0.7, do_sample=True):
        """
        基于图像生成回答
        
        Args:
            image_list_list: 图像列表的列表，每个子列表包含一个样本的所有图像
            question_list: 问题列表
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否使用采样
            
        Returns:
            生成的回答列表
        """
        if len(image_list_list) != len(question_list):
            raise ValueError("图像列表和问题列表长度不匹配")
        
        results = []
        
        for images, question in zip(image_list_list, question_list):
            try:
                # 处理图像
                if len(images) == 0:
                    print("警告: 空图像列表，跳过此样本")
                    results.append("无图像输入")
                    continue
                
                # 取第一张图像（RoboBrain主要处理单图像）
                image = self.preprocess_image(images[0])
                
                # 构建消息格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image", "image": image},
                        ],
                    },
                ]
                
                # 处理输入
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # 移动到设备 - 多卡情况下不需要手动移动
                if not self.is_model_parallel:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    # 多卡情况下，输入数据移动到第一个可用设备
                    first_device = next(iter(self.model.hf_device_map.values()))
                    inputs = {k: v.to(first_device) for k, v in inputs.items()}
                
                # 生成回答
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # 解码输出
                prediction = self.processor.decode(
                    output[0][2:],
                    skip_special_tokens=True
                ).split("assistant")[-1].strip()
                
                results.append(prediction)
                
            except Exception as e:
                print(f"处理样本时出错: {e}")
                results.append(f"处理错误: {str(e)}")
                
        return results
    
    def generate_clip(self, video_list_list, question_list, max_new_tokens=512, temperature=0.7, do_sample=True):
        """
        基于视频/图像序列生成回答
        
        Args:
            video_list_list: 视频帧列表的列表
            question_list: 问题列表
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否使用采样
            
        Returns:
            生成的回答列表
        """
        # RoboBrain主要处理图像，对于视频我们取关键帧
        image_list_list = []
        for video_frames in video_list_list:
            if len(video_frames) > 0:
                # 取中间帧作为代表
                mid_frame = video_frames[len(video_frames) // 2]
                image_list_list.append([mid_frame])
            else:
                image_list_list.append([])
        
        return self.generate_image(image_list_list, question_list, max_new_tokens, temperature, do_sample)
    
    def move_to_device(self, device):
        """移动模型到指定设备"""
        if self.model is not None and device != self.device:
            if self.is_model_parallel:
                print("⚠️  警告: 模型已分布到多个GPU，无法移动到单个设备")
                print("如需切换到单卡模式，请重新初始化模型")
            else:
                self.model = self.model.to(device)
                self.device = device
                print(f"模型已移动到设备: {device}")
    
    def get_gpu_info(self):
        """获取GPU信息"""
        if not torch.cuda.is_available():
            return "无可用GPU"
        
        info = []
        info.append(f"GPU数量: {self.gpu_count}")
        
        for i in range(self.gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info.append(f"GPU {i}: {name} ({memory:.1f}GB)")
        
        if self.is_model_parallel:
            info.append("模式: 多卡并行")
            if hasattr(self.model, 'hf_device_map'):
                info.append("设备分配:")
                for layer, device in list(self.model.hf_device_map.items())[:5]:
                    info.append(f"  {layer}: {device}")
                if len(self.model.hf_device_map) > 5:
                    info.append(f"  ... 还有 {len(self.model.hf_device_map) - 5} 层")
        else:
            info.append(f"模式: 单卡 ({self.device})")
        
        return "\n".join(info)
    
    def clear_cache(self):
        """清理缓存"""
        if torch.cuda.is_available():
            # 清理所有GPU的缓存
            for i in range(self.gpu_count):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()
        print("缓存已清理")
    
    def __del__(self):
        """析构函数，清理资源"""
        self.clear_cache() 