import os
import math
import torch
from PIL import Image
from io import BytesIO
import requests
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from peng_utils import get_image, contact_img

def split_magma_model(model_path):
    """Magma模型多GPU分布函数"""
    device_map = {}
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    if world_size <= 1:
        return None  # 单GPU或CPU，使用默认加载
    
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.num_hidden_layers
        print(f"Magma model has {num_layers} layers")
        
        # 使用所有可用GPU分布模型层
        available_gpus = list(range(world_size))
        num_layers_per_gpu = math.ceil(num_layers / len(available_gpus))
        
        # 分布语言模型层
        layer_cnt = 0
        for i, gpu_id in enumerate(available_gpus):
            start_layer = i * num_layers_per_gpu
            end_layer = min((i + 1) * num_layers_per_gpu, num_layers)
            for layer_idx in range(start_layer, end_layer):
                if layer_cnt < num_layers:
                    device_map[f'model.layers.{layer_cnt}'] = gpu_id
                    layer_cnt += 1
        
        # 将关键组件放到GPU 0（通常有最多可用内存）
        device_map['model.embed_tokens'] = 0
        device_map['model.norm'] = 0
        device_map['lm_head'] = 0
        
        # Vision相关组件也放到GPU 0
        device_map['vision_model'] = 0
        device_map['multi_modal_projector'] = 0
        
        # 如果有rotary embedding，也放到GPU 0
        if hasattr(config, 'rope_theta'):
            device_map['model.rotary_emb'] = 0
        
        print(f"Device map created for GPUs {available_gpus} (using GPU 0 as main)")
        print("Vision model and embeddings on GPU 0")
        return device_map
        
    except Exception as e:
        print(f"Failed to create device map: {e}")
        return None

def setup_device_config(device=None):
    """设置设备配置和内存优化选项"""
    config = {
        "device": "cuda:0",
        "device_map": None,
        "load_8bit": False,
        "torch_dtype": torch.bfloat16,
    }
    
    if not torch.cuda.is_available():
        config["device"] = "cpu"
        config["torch_dtype"] = torch.float32
        return config
    
    if device is not None:
        # 解析设备参数
        if isinstance(device, (int, str)):
            if str(device).isdigit():
                config["device"] = f"cuda:{device}"
            elif 'cuda:' in str(device):
                config["device"] = str(device)
            else:
                config["device"] = device
    
    # 检查多GPU情况
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"Multiple GPUs detected ({gpu_count}), enabling multi-GPU mode")
        config["device_map"] = "auto"  # 让transformers自动分配
        config["device"] = "cuda:0"  # 主设备
    else:
        device_id = int(config["device"].split(':')[-1]) if 'cuda:' in config["device"] else 0
        try:
            # 获取GPU内存信息
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            print(f"GPU {device_id} total memory: {gpu_memory_gb:.1f} GB")
            
            # 如果GPU内存不足，启用内存优化
            if gpu_memory_gb < 16:  # Magma-8B需要约16GB内存
                print("GPU memory insufficient, enabling memory optimizations...")
                config["load_8bit"] = True
                config["device_map"] = "auto"  # 允许CPU offload
                print("Enabled: 8-bit quantization and CPU offloading")
            else:
                config["device_map"] = {"": config["device"]}
                
        except Exception as e:
            print(f"Could not check GPU memory, using safe defaults: {e}")
            config["load_8bit"] = True
            config["device_map"] = "auto"
    
    return config

class TestMagma:
    """Magma-8B模型测试类，支持多GPU并行推理"""
    
    def __init__(self, device=None):
        # 设置模型路径
        self.model_path = "/home/fx/Exp2/test/EmbodiedEval/msjeval/Magma-8B"
        
        # 设置环境变量优化GPU内存使用
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 异步执行
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer并行警告
        
        print(f"Initializing Magma model from: {self.model_path}")
        
        try:
            # 配置设备和内存优化
            self.device_config = setup_device_config(device)
            self.device = self.device_config["device"]
            self.dtype = self.device_config["torch_dtype"]
            
            print(f"Device config: {self.device_config}")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 根据GPU数量选择加载策略
            gpu_count = torch.cuda.device_count()
            print(f"Available GPUs: {gpu_count}")
            
            if gpu_count > 1:
                print(f"Loading Magma model with multi-GPU support ({gpu_count} GPUs)")
                # 强制使用device_map="auto"进行多卡分配
                self.device_config["device_map"] = "auto"
                
                # 加载模型和processor - 多GPU版本
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=self.dtype,
                    device_map="auto",  # 强制多卡分配
                    low_cpu_mem_usage=True,
                )
            else:
                print(f"Loading Magma model on single device: {self.device}")
                # 单GPU或CPU版本
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=self.dtype,
                    device_map=self.device_config["device_map"],
                    load_in_8bit=self.device_config["load_8bit"],
                )
                
                # 只有在非device_map模式下才手动移动模型
                if self.device_config["device_map"] is None:
                    self.model.to(self.device)
            
            # 加载processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            print("Magma model and processor loaded successfully")
            print(f"Model dtype: {self.dtype}, Device: {self.device}")
            if gpu_count > 1:
                print(f"Multi-GPU mode enabled with {gpu_count} GPUs")
                if hasattr(self.model, 'hf_device_map'):
                    print(f"Device map: {self.model.hf_device_map}")
            
        except Exception as e:
            print(f"Error loading Magma model: {e}")
            raise e
    
    def move_to_device(self, device=None):
        """移动模型到指定设备（多GPU模式下有限制）"""
        gpu_count = torch.cuda.device_count()
        
        if gpu_count > 1 and hasattr(self.model, 'hf_device_map'):
            print("Model is distributed across multiple GPUs, cannot move to single device")
            print(f"Current device map: {self.model.hf_device_map}")
            return
        
        if device is not None and hasattr(device, 'type') and 'cuda' in device.type:
            self.dtype = torch.bfloat16
            self.device = device
            if not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(device)
            print(f"Model moved to GPU: {device}")
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            if not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to('cpu')
            print(f"Model moved to CPU")
    
    def generate_image(self, image, question):
        """使用Magma模型生成图像描述或回答问题"""
        print("Magma generation begins")
        print(f"image: {type(image)}\nquestion: {type(question)} {question}")
        
        outputs = []
        image_list_list = image
        question_list = question
        
        for image_list, question in zip(image_list_list, question_list):
            try:
                # 处理图像
                if len(image_list) > 0:
                    # 取第一张图像
                    img = image_list[0]
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img.astype('uint8')).convert('RGB')
                    elif not isinstance(img, Image.Image):
                        img = get_image(img)
                    
                    # 设置对话格式
                    convs = [
                        {"role": "system", "content": "You are agent that can see, talk and act."},
                        {"role": "user", "content": f"<image_start><image><image_end>\n{question}"},
                    ]
                    
                    # 应用聊天模板
                    prompt = self.processor.tokenizer.apply_chat_template(
                        convs, tokenize=False, add_generation_prompt=True
                    )
                    
                    # 处理输入
                    inputs = self.processor(images=[img], texts=prompt, return_tensors="pt")
                    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
                    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
                    
                    # 多GPU模式下，输入应该发送到主设备（通常是cuda:0）
                    target_device = self.device
                    if hasattr(self.model, 'hf_device_map'):
                        # 在多GPU模式下，输入通常发送到第一个设备
                        target_device = 'cuda:0'
                    
                    inputs = inputs.to(target_device)
                    # 只对非整数张量应用dtype转换
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                            inputs[key] = value.to(self.dtype)
                    
                    # 生成配置
                    generation_args = { 
                        "max_new_tokens": 128, 
                        "temperature": 0.0, 
                        "do_sample": False, 
                        "use_cache": True,
                        "num_beams": 1,
                    }
                    
                    # 进行推理
                    with torch.inference_mode():
                        generate_ids = self.model.generate(**inputs, **generation_args)
                    
                    # 解码输出
                    generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
                    response = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
                    
                    outputs.append(response)
                    print(f"Generated response: {response}")
                
                else:
                    # 没有图像的情况
                    outputs.append("No image provided")
                    
            except Exception as e:
                print(f"Error during generation: {e}")
                outputs.append(f"Error: {str(e)}")
        
        return outputs
    
    def chat_image(self, image, question):
        """简化的聊天接口"""
        if isinstance(image, (list, tuple)):
            if len(image) > 0:
                img = image[0]
            else:
                return "No image provided"
        else:
            img = image
            
        # 将单个图像和问题转换为列表格式
        return self.generate_image([[img]], [question])[0]
    
    def test_with_url(self, url=None, question="What is in this image?"):
        """使用URL测试模型"""
        if url is None:
            url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png"
        
        try:
            image = Image.open(BytesIO(requests.get(url, stream=True).content))
            image = image.convert("RGB")
            return self.chat_image(image, question)
        except Exception as e:
            return f"Error loading image from URL: {e}"
    
    def get_memory_usage(self):
        """获取GPU内存使用情况"""
        if not torch.cuda.is_available():
            return "CUDA not available"
        
        memory_info = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_info.append(f"GPU {i}: {allocated:.1f}GB/{total:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        return "\n".join(memory_info)
    
    def get_device_info(self):
        """获取模型设备分布信息"""
        info = {
            "device": self.device,
            "dtype": self.dtype,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_map": getattr(self.model, 'hf_device_map', None),
            "is_multi_gpu": hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None
        }
        return info

# 为了兼容性，添加一个别名
TestMagma8B = TestMagma 