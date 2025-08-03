import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
from typing import List, Union, Optional

# 添加SAT和LLaVA路径
sat_path = "/home/fx/Exp2/test/EmbodiedEval/SAT"
llava_path = os.path.join(sat_path, "models/LLaVA_modified/LLaVA")
sys.path.append(sat_path)
sys.path.append(llava_path)

# 尝试导入peft，如果版本不兼容则使用替代方案
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PEFT import failed: {e}")
    PEFT_AVAILABLE = False
    # 尝试降级处理
    try:
        import subprocess
        import pkg_resources
        
        # 检查当前版本
        current_accelerate = pkg_resources.get_distribution("accelerate").version
        current_peft = pkg_resources.get_distribution("peft").version
        
        print(f"Current versions - accelerate: {current_accelerate}, peft: {current_peft}")
        
        # 建议的兼容版本
        print("Trying to install compatible versions...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate==0.21.0", "peft==0.4.0"])
        
        # 重新导入
        from peft import PeftModel
        PEFT_AVAILABLE = True
        print("Successfully installed compatible versions")
        
    except Exception as install_error:
        print(f"Failed to install compatible versions: {install_error}")
        PEFT_AVAILABLE = False

from huggingface_hub import hf_hub_download

# 现在尝试导入SAT模型
try:
    import models.model_interface as models
    print("Successfully imported SAT models")
except ImportError as e:
    print(f"Failed to import SAT models: {e}")
    print("Please ensure LLaVA is properly installed in the SAT directory")
    raise

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    expand2square,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

class TestSAT:
    """SAT模型推理类，支持多卡推理和多模态对话"""
    
    def __init__(self, device=None):
        """
        初始化SAT模型
        
        Args:
            device: 设备参数（兼容peng_utils接口）
        """
        self.device = device
        self.dtype = torch.float32  # 使用float32避免Half精度问题
        self.temperature = 0
        self.top_p = 1.0
        self.num_beams = 1
        self.max_new_tokens = 768
        self.checkpoint_name = None  # 如果有LoRA路径可设置
        self._init_model()
    
    def _init_model(self):
        """初始化SAT模型"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # 设置更激进的内存优化
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                
                # 设置更保守的内存分配策略
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
                
                # 检查GPU内存使用情况
                gpu_count = torch.cuda.device_count()
                print(f"🔍 检测到 {gpu_count} 个GPU")
                for i in range(gpu_count):
                    total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
                    free_memory = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3
                    print(f"GPU {i}: 总内存: {total_memory:.2f}GB, 已分配: {allocated_memory:.2f}GB, 可用: {free_memory:.2f}GB")
            
            # 尝试导入量化相关库
            try:
                import transformers
                from transformers import BitsAndBytesConfig
                QUANTIZATION_AVAILABLE = True
                print("Quantization libraries available")
            except ImportError:
                QUANTIZATION_AVAILABLE = False
                print("Quantization libraries not available, falling back to FP16")
            
            gpu_count = torch.cuda.device_count()
            print(f"Available GPUs: {gpu_count}")
            config = {
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'num_beams': self.num_beams,
                'max_new_tokens': self.max_new_tokens,
                'output_hidden_states': False
            }
            from SAT.models import model_interface as models
            from transformers import AutoTokenizer
            from transformers import CLIPImageProcessor
            global PEFT_AVAILABLE
            try:
                from peft import PeftModel
                PEFT_AVAILABLE = True
            except ImportError:
                PEFT_AVAILABLE = False
            
            # 多卡 - 使用更保守的策略
            if gpu_count > 1:
                print(f"Multiple GPUs detected ({gpu_count}), using conservative single-GPU mode")
                print("Loading base model on single GPU to avoid memory issues...")
                
                # 强制使用单GPU模式，避免内存不足
                device_map = "cuda:0"  # 只使用第一个GPU
                
                # 设置4bit量化配置（优先）或8bit量化配置
                if QUANTIZATION_AVAILABLE:
                    try:
                        print("Attempting 4bit quantization for maximum memory efficiency...")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                        )
                        print("4bit quantization config created successfully")
                    except Exception as e:
                        print(f"4bit quantization failed: {e}, falling back to 8bit...")
                        try:
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0,
                                llm_int8_has_fp16_weight=False,
                            )
                            print("8bit quantization config created successfully")
                        except Exception as e2:
                            print(f"8bit quantization also failed: {e2}, using FP16")
                            quantization_config = None
                else:
                    quantization_config = None
                    print("Using FP16 precision (no quantization)")
                
                # 使用单GPU加载
                base_model = models.LlavaModel_13B_Interface(config, device_map=device_map)
            
                # 应用量化到模型
                if QUANTIZATION_AVAILABLE and quantization_config is not None:
                    print("Applying quantization to model...")
                    try:
                        base_model.model = base_model.model.quantize(quantization_config)
                        print("Quantization applied successfully")
                    except Exception as e:
                        print(f"Quantization application failed: {e}, continuing with unquantized model")
                
                if PEFT_AVAILABLE and self.checkpoint_name:
                    print("Loading LoRA model...")
                    try:
                        lora_model = PeftModel.from_pretrained(base_model, self.checkpoint_name)
                        # 保持量化精度
                        if QUANTIZATION_AVAILABLE:
                            lora_model = lora_model.half()
                        print("Merging and unloading LoRA...")
                        self.model = lora_model.merge_and_unload()
                    except Exception as lora_error:
                        print(f"LoRA loading failed: {lora_error}")
                        print("Falling back to base model...")
                        self.model = base_model
                else:
                    self.model = base_model
                
                # 强制模型使用float32避免Half精度问题
                try:
                    self.model = self.model.float()
                    print("✅ 模型已转换为float32精度")
                except Exception as e:
                    print(f"⚠️ 模型精度转换失败: {e}")
                
                print("SAT model loaded successfully with multi-GPU support")
            else:
                print(f"Loading SAT model on single device: {self.device}")
                base_model = models.LlavaModel_13B_Interface(config, device_map=None)
            
                # 单卡也应用量化
                if QUANTIZATION_AVAILABLE:
                    try:
                        print("Attempting 4bit quantization for single device...")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                        )
                        base_model.model = base_model.model.quantize(quantization_config)
                        print("4bit quantization applied to single device model")
                    except Exception as e:
                        print(f"4bit quantization failed: {e}, trying 8bit...")
                        try:
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0,
                                llm_int8_has_fp16_weight=False,
                            )
                            base_model.model = base_model.model.quantize(quantization_config)
                            print("8bit quantization applied to single device model")
                        except Exception as e2:
                            print(f"8bit quantization also failed: {e2}, using unquantized model")
                
                if PEFT_AVAILABLE and self.checkpoint_name:
                    print("Loading LoRA model...")
                    try:
                        lora_model = PeftModel.from_pretrained(base_model, self.checkpoint_name)
                        if QUANTIZATION_AVAILABLE:
                            lora_model = lora_model.half()
                        print("Merging and unloading LoRA...")
                        self.model = lora_model.merge_and_unload()
                    except Exception as lora_error:
                        print(f"LoRA loading failed: {lora_error}")
                        print("Falling back to base model...")
                        self.model = base_model
                else:
                    self.model = base_model
                
                # 强制模型使用float32避免Half精度问题
                try:
                    self.model = self.model.float()
                    print("✅ 模型已转换为float32精度")
                except Exception as e:
                    print(f"⚠️ 模型精度转换失败: {e}")
                
                print("SAT model loaded successfully on single device")
            self.tokenizer = base_model.tokenizer
            self.image_processor = base_model.image_processor
            
            # 最终内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"Error loading SAT model: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def move_to_device(self, device):
        # 调试信息
        print(f"move_to_device called with device: {device}")
        print(f"Model type: {type(self.model)}")
        print(f"Model has hf_device_map: {hasattr(self.model, 'hf_device_map')}")
        if hasattr(self.model, 'hf_device_map'):
            print(f"hf_device_map value: {self.model.hf_device_map}")
        
        # 多卡模式下不迁移
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
            print("Model is distributed across multiple GPUs, cannot move to single device")
            print(f"Current device map: {self.model.hf_device_map}")
            return
        
        # 兼容int/str/torch.device
        if isinstance(device, torch.device):
            device_str = device.type if device.index is None else f"{device.type}:{device.index}"
        else:
            device_str = str(device)
        if "cuda" in device_str:
            # 使用FP32避免Half精度问题
            self.dtype = torch.float32
            self.device = device_str
            try:
                # 强制模型使用float32
                self.model = self.model.float()
                self.model = self.model.to(device_str)
                print(f"Model moved to GPU: {device_str} with dtype: {self.dtype}")
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM when moving model to {device_str}: {e}")
                print("Model may already be on correct device or using quantization")
                # 不抛出异常，继续使用
        else:
            self.dtype = torch.float32
            self.device = "cpu"
            try:
                self.model = self.model.to("cpu")
                print("Model moved to CPU")
            except Exception as e:
                print(f"Error moving model to CPU: {e}")
                print("Model may already be on correct device")
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """加载图像为PIL图像"""
        try:
            if isinstance(image_input, str):
                # 文件路径
                if os.path.exists(image_input):
                    return Image.open(image_input).convert('RGB')
                else:
                    raise FileNotFoundError(f"Image file not found: {image_input}")
            elif isinstance(image_input, np.ndarray):
                # numpy数组
                return Image.fromarray(image_input.astype('uint8')).convert('RGB')
            elif isinstance(image_input, Image.Image):
                # PIL图像
                return image_input.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
    
    def _preprocess_images(self, images: List[Union[str, np.ndarray, Image.Image]]) -> torch.Tensor:
        """预处理图像列表"""
        processed_images = []
        
        def flatten_image_list(image_input):
            """将任意格式的图像输入展平为PIL.Image列表"""
            def _recursive_flatten(item, depth=0, max_depth=10):
                if depth > max_depth:
                    return []
                    
                if isinstance(item, list):
                    flat_images = []
                    for sub_item in item:
                        flat_images.extend(_recursive_flatten(sub_item, depth + 1, max_depth))
                    return flat_images
                elif isinstance(item, (int, float)):
                    return []
                else:
                    try:
                        loaded_img = self._load_image(item)
                        if loaded_img is not None:
                            return [loaded_img]
                        else:
                            return []
                    except Exception as e:
                        # 减少调试输出
                        return []
            
            return _recursive_flatten(image_input)
        
        # 展平输入图像列表
        flat_images = flatten_image_list(images)
        
        if not flat_images:
            print("No valid images found, creating default black image")
            # 创建一个默认的黑色图像
            default_image = Image.new('RGB', (224, 224), (0, 0, 0))
            flat_images = [default_image]
        
        for pil_image in flat_images:
            try:
                # 扩展为正方形
                image = expand2square(pil_image, tuple(int(x*255) for x in self.image_processor.image_mean))
                
                # 预处理
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                processed_images.append(image)
            except Exception as e:
                print(f"Error preprocessing image: {e}")
                # 创建一个默认的黑色图像作为fallback
                default_image = Image.new('RGB', (224, 224), (0, 0, 0))
                image = expand2square(default_image, tuple(int(x*255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                processed_images.append(image)
        
        # 堆叠为batch
        pixel_values = torch.stack(processed_images, dim=0)
        return pixel_values  # 不要 .to('cuda')
    
    def _preprocess_text(self, prompt: str) -> tuple:
        """预处理文本"""
        input_ids = []
        attention_mask = []
        
        # 添加图像token
        input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids.append(input_id)
        attention_mask.append(torch.ones_like(input_id))
        
        # 填充序列
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return input_ids, attention_mask  # 不要 .to('cuda')
    
    def chat_image(self, 
                   image: Union[str, np.ndarray, Image.Image, List], 
                   question: str) -> str:
        """
        与图像进行对话
        
        Args:
            image: 图像输入（可以是路径、numpy数组、PIL图像或列表）
            question: 问题文本
        
        Returns:
            模型回答
        """
        try:
            # 处理单个图像或图像列表
            if isinstance(image, list):
                return self._chat_multiple_images(image, question)
            else:
                return self._chat_single_image(image, question)
        except Exception as e:
            print(f"Error in chat_image: {e}")
            return f"Error: {str(e)}"
    
    def _chat_single_image(self, image: Union[str, np.ndarray, Image.Image], question: str) -> str:
        try:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            pixel_values = self._preprocess_images([image])
            input_ids, attention_mask = self._preprocess_text(question)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
            }
            
            # 确保输入使用正确的dtype，避免Half精度问题
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                    # 使用float32避免Half精度问题
                    inputs[key] = value.to(torch.float32)
            
            # 使用更保守的推理设置
            with torch.no_grad():
                try:
                    # 确保所有输入都在正确的设备上
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            # 获取模型设备
                            model_device = next(self.model.parameters()).device
                            inputs[key] = value.to(model_device)
                    
                    # SAT模型的forward方法不接受生成参数，直接使用inputs
                    generated_ids = self.model(**inputs)
                    generated_ids[generated_ids == -200] = 1
                    generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    return generated_text[0] if generated_text else ""
                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA OOM during inference: {e}")
                    # 尝试清理内存并重试
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    
                    # 使用更小的batch size重试
                    try:
                        # 如果有多张图像，只使用第一张
                        if inputs['pixel_values'].shape[0] > 1:
                            inputs['pixel_values'] = inputs['pixel_values'][:1]
                            inputs['input_ids'] = inputs['input_ids'][:1]
                            inputs['attention_mask'] = inputs['attention_mask'][:1]
                        
                        # 重试时也直接使用inputs
                        generated_ids = self.model(**inputs)
                        generated_ids[generated_ids == -200] = 1
                        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        return generated_text[0] if generated_text else ""
                    except Exception as e2:
                        print(f"Retry also failed: {e2}")
                        return f"Error: Memory insufficient for inference - {e2}"
                        
        except Exception as e:
            print(f"Error in _chat_single_image: {e}")
            return f"Error: {str(e)}"
    
    def _chat_multiple_images(self, images: List[Union[str, np.ndarray, Image.Image]], question: str) -> str:
        pixel_values = self._preprocess_images(images)
        input_ids, attention_mask = self._preprocess_text(question)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
        }
        
        # 确保输入使用正确的dtype，避免Half精度问题
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                # 使用float32避免Half精度问题
                inputs[key] = value.to(torch.float32)
        
        with torch.no_grad():
            # 确保所有输入都在正确的设备上
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    # 获取模型设备
                    model_device = next(self.model.parameters()).device
                    inputs[key] = value.to(model_device)
            
            # SAT模型的forward方法不接受生成参数，直接使用inputs
            generated_ids = self.model(**inputs)
            generated_ids[generated_ids == -200] = 1
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text[0] if generated_text else ""
    
    def generate_image(self, image, question):
        """生成图像回答（兼容peng_utils接口）"""
        print("generate begins")
        print(f"image:{type(image)}\nquestion:{type(question)}{question}")
        
        outputs = []
        image_list_list = image
        question_list = question
        
        for image_list, question in zip(image_list_list, question_list):
            try:
                # 处理图像列表，取第一张图像
                if len(image_list) > 0:
                    img = image_list[0]
                    # 直接调用chat_image处理，不通过get_image
                    output = self.chat_image(img, question)
                    outputs.append(output)
                else:
                    outputs.append("No image provided")
                    
            except Exception as e:
                print(f"Error processing image: {e}")
                outputs.append(f"Error: {str(e)}")
        
        print("generate finish")
        return outputs
    
    def batch_generate(self, image_list: List[Union[str, np.ndarray, Image.Image]], 
                      question_list: List[str]) -> List[str]:
        """
        批量生成回答
        
        Args:
            image_list: 图像列表
            question_list: 问题列表
        
        Returns:
            回答列表
        """
        if len(image_list) != len(question_list):
            raise ValueError("Image list and question list must have the same length")
        
        answers = []
        for i, (image, question) in enumerate(zip(image_list, question_list)):
            try:
                answer = self.generate_image(image, question)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                answers.append(f"Error: {str(e)}")
        
        return answers


def test_sat():
    """测试SAT模型"""
    print("=== Testing SAT Model with 4bit Quantization ===")
    
    # 检查GPU内存
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            free_memory = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Total: {total_memory:.2f}GB, Allocated: {allocated_memory:.2f}GB, Reserved: {free_memory:.2f}GB")
    
    # 初始化模型（使用多卡）
    try:
        model = TestSAT(
            device="cuda:0" # 指定设备
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查模型是否成功加载并量化
    print("\n--- Model Status Check ---")
    print(f"Model type: {type(model.model)}")
    if hasattr(model.model, 'hf_device_map'):
        print(f"Device map: {model.model.hf_device_map}")
    
    # 检查模型参数是否量化
    total_params = 0
    quantized_params = 0
    for name, param in model.model.named_parameters():
        total_params += param.numel()
        if hasattr(param, 'dtype'):
            if param.dtype in [torch.int8, torch.uint8]:
                quantized_params += param.numel()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Quantized parameters: {quantized_params:,}")
    if total_params > 0:
        quantization_ratio = quantized_params / total_params * 100
        print(f"Quantization ratio: {quantization_ratio:.2f}%")
    
    # 创建测试图像
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # 灰色图像
    
    # 测试单个图像
    print("\n--- Single Image Test ---")
    question = "What color is this image?"
    print(f"Question: {question}")
    
    try:
        answer = model.generate_image(test_image, question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试多个图像
    print("\n--- Multiple Images Test ---")
    test_images = [
        np.ones((224, 224, 3), dtype=np.uint8) * 128,  # 灰色
        np.ones((224, 224, 3), dtype=np.uint8) * 255,  # 白色
    ]
    question = "What are the colors of these images?"
    print(f"Question: {question}")
    
    try:
        answer = model.generate_image(test_images, question)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试批量生成
    print("\n--- Batch Generation Test ---")
    questions = ["What color is this?", "Is this image bright?"]
    print(f"Questions: {questions}")
    
    try:
        answers = model.batch_generate(test_images, questions)
        for i, answer in enumerate(answers):
            print(f"Answer {i+1}: {answer}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_sat() 