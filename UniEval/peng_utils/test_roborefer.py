import os
import sys
import cv2
from typing import List, Dict, Any, Optional, Union
import torch
from PIL import Image
import numpy as np

# 禁用Flash Attention避免安装依赖
os.environ["_FLASH_ATTN_AVAILABLE"] = "False"
os.environ["FLASH_ATTN_AVAILABLE"] = "False"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Add the RoboRefer project to the python path to import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboRefer'))

# 在导入 transformers 相关模块之前，禁用某些自动注册
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except:
    pass

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images, KeywordsStoppingCriteria
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava import conversation as clib

# 全局模型变量，避免重复加载
_global_model = None
_global_tokenizer = None
_global_image_processor = None
_global_conv = None

def load_roborefer_model(model_path: str = "/home/fx/Exp2/test/EmbodiedEval/msjeval/RoboRefer-2B-SFT",
                        device: str = "cuda:0",
                        torch_dtype: torch.dtype = torch.bfloat16):
    """
    加载RoboRefer模型（全局单例模式）。
    
    Args:
        model_path: RoboRefer模型 checkpoints 路径。
        device: 主设备。
        torch_dtype: 模型权重的数据类型。
    """
    global _global_model, _global_tokenizer, _global_image_processor, _global_conv
    
    if _global_model is not None:
        return _global_model, _global_tokenizer, _global_image_processor, _global_conv
    
    print(f"Loading RoboRefer model from {model_path}")
    
    model_name = get_model_name_from_path(model_path)
    
    # 使用项目自带的加载器，单卡模式
    _global_tokenizer, _global_model, _global_image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_name=model_name,
        model_base=None,
        device_map=None,  # 单卡模式
        torch_dtype=torch_dtype,
    )
    
    # 手动移动到指定设备
    if device != "cpu":
        _global_model = _global_model.to(device)

    # 根据模型配置选择对话模板
    conv_template_name = "v1" # 默认模板
    if 'qwen2' in model_name.lower():
        conv_template_name = "qwen_1_5"
    elif 'mistral' in model_name.lower() or 'mixtral' in model_name.lower():
        conv_template_name = 'mistral_instruct'
    elif 'llama' in model_name.lower():
        conv_template_name = "llama_3"

    _global_conv = clib.conv_templates[conv_template_name].copy()
    
    print("RoboRefer model loaded successfully!")
    
    return _global_model, _global_tokenizer, _global_image_processor, _global_conv

def load_single_image(image_input: Union[str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
    """
    加载单张图像为PIL.Image对象。
    
    Args:
        image_input: 图像输入，可以是路径、numpy数组或PIL图像。
    
    Returns:
        PIL.Image对象，如果加载失败返回None。
    """
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                print(f"Warning: Image file not found: {image_input}")
                return None
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # 处理numpy数组
            if len(image_input.shape) == 3:
                if image_input.shape[2] == 3:
                    # RGB图像
                return Image.fromarray(image_input.astype('uint8')).convert('RGB')
                elif image_input.shape[2] == 1:
                    # 单通道图像，转为RGB
                    return Image.fromarray(image_input[:, :, 0].astype('uint8')).convert('RGB')
                else:
                    # 其他通道数，取前3个通道
                    return Image.fromarray(image_input[:, :, :3].astype('uint8')).convert('RGB')
            elif len(image_input.shape) == 2:
                # 灰度图像，转为RGB
                return Image.fromarray(image_input.astype('uint8')).convert('RGB')
            else:
                print(f"Warning: Unexpected numpy array shape: {image_input.shape}")
                return None
        elif isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        elif isinstance(image_input, (int, float)):
            # 数字类型，跳过
            return None
        else:
            print(f"Warning: Unsupported image type: {type(image_input)}")
            return None
    except Exception as e:
        print(f"Warning: Failed to load image {type(image_input)}: {e}")
        return None

def flatten_image_list(image_input: Union[str, np.ndarray, Image.Image, List]) -> List[Image.Image]:
    """
    将任意格式的图像输入展平为PIL.Image列表。
    优先处理图像路径列表（benchmark标准格式）。
    
    Args:
        image_input: 图像输入，支持嵌套列表、路径列表等。
    
    Returns:
        PIL.Image对象列表。
    """
    def _recursive_flatten(item, depth=0, max_depth=10):
        if depth > max_depth:
            return []
            
        # 只在深度0和1时打印调试信息，避免过多输出
        if depth <= 1:
            print(f"Debug: Processing item at depth {depth}, type: {type(item)}")
            
        if isinstance(item, list):
            if depth <= 1:
                print(f"Debug: List at depth {depth} has {len(item)} items")
            
            # 检查是否可能是图像路径列表（benchmark格式）
            if depth == 0 and all(isinstance(x, str) for x in item):
                print(f"Debug: Detected image path list with {len(item)} paths")
                # 直接处理图像路径列表
                images = []
                for path in item:
                    img = load_single_image(path)
                    if img is not None:
                        images.append(img)
                    else:
                        print(f"Warning: Failed to load image from path: {path}")
                return images
            
            # 检查是否包含numpy数组（可能是图像数据）
            has_numpy = any(isinstance(x, np.ndarray) for x in item)
            if has_numpy:
                print(f"Debug: Found numpy arrays in list at depth {depth}")
                images = []
                for sub_item in item:
                    if isinstance(sub_item, np.ndarray):
                        if depth <= 2:
                            print(f"Debug: Processing numpy array with shape: {sub_item.shape}")
                        img = load_single_image(sub_item)
                        if img is not None:
                            images.append(img)
                        else:
                            print(f"Warning: Failed to load numpy array with shape: {sub_item.shape}")
                    else:
                        # 递归处理非numpy数组
                        images.extend(_recursive_flatten(sub_item, depth + 1, max_depth))
                return images
            
            # 检查是否包含大量数字（可能是像素数据）
            if depth >= 2:
                # 在深层嵌套中，如果遇到大量数字，尝试重建图像
                all_numbers = []
                def extract_numbers_recursive(lst, current_depth=0):
                    if current_depth > 3:  # 限制递归深度
                        return
                    for elem in lst:
                        if isinstance(elem, (int, float)):
                            all_numbers.append(elem)
                        elif isinstance(elem, list) and current_depth < 3:
                            extract_numbers_recursive(elem, current_depth + 1)
                
                extract_numbers_recursive(item)
                
                if len(all_numbers) > 1000:  # 足够的数据重建图像
                    print(f"Debug: Found {len(all_numbers)} numbers at depth {depth}, attempting to create image")
                    try:
                        # 尝试创建图像
                        import math
                        # 假设是RGB数据
                        if len(all_numbers) % 3 == 0:
                            pixels = len(all_numbers) // 3
                            size = int(math.sqrt(pixels))
                            if size * size == pixels:
                                # 创建RGB图像
                                img_array = np.array(all_numbers[:pixels*3]).reshape(size, size, 3).astype(np.uint8)
                            pil_img = Image.fromarray(img_array).convert('RGB')
                                print(f"Debug: Successfully created {size}x{size}x3 image from {len(all_numbers)} numbers")
                            return [pil_img]
                        elif len(all_numbers) > 1000:
                            # 尝试创建灰度图像
                            size = int(math.sqrt(len(all_numbers)))
                            if size * size == len(all_numbers):
                                img_array = np.array(all_numbers[:size*size]).reshape(size, size).astype(np.uint8)
                                pil_img = Image.fromarray(img_array).convert('RGB')
                                print(f"Debug: Successfully created {size}x{size} grayscale image from {len(all_numbers)} numbers")
                                return [pil_img]
                except Exception as e:
                        print(f"Debug: Failed to create image from numbers: {e}")
            
            # 递归处理嵌套列表（限制深度以避免过度递归）
            if depth < 3:
            flat_images = []
                for i, sub_item in enumerate(item):
                    if depth <= 1:
                        print(f"Debug: Processing sub-item {i} at depth {depth}")
                flat_images.extend(_recursive_flatten(sub_item, depth + 1, max_depth))
            return flat_images
            else:
                # 在深层嵌套中，跳过处理以避免过度递归
                return []
                
        elif isinstance(item, (int, float)):
            # 数字类型，跳过（不在深层打印）
            if depth <= 2:
                print(f"Debug: Skipping numeric value: {item}")
            return []
        elif isinstance(item, np.ndarray):
            if depth <= 2:
                print(f"Debug: Processing numpy array with shape: {item.shape}")
            loaded_img = load_single_image(item)
            if loaded_img is not None:
                return [loaded_img]
            else:
                print(f"Warning: Failed to load numpy array with shape: {item.shape}")
            return []
        else:
            if depth <= 2:
                print(f"Debug: Processing other type: {type(item)}")
            loaded_img = load_single_image(item)
            if loaded_img is not None:
                return [loaded_img]
            else:
                if depth <= 2:
                    print(f"Warning: Failed to load image at depth {depth}, type: {type(item)}")
                return []
    
    return _recursive_flatten(image_input)

def query_roborefer(image: Union[str, np.ndarray, Image.Image, List], 
                   question: str,
                   model_path: str = "/home/fx/Exp2/test/EmbodiedEval/msjeval/RoboRefer-2B-SFT",
                   device: str = "cuda:0",
                   torch_dtype: torch.dtype = torch.bfloat16,
                   max_new_tokens: int = 512,
                   temperature: float = 0.2) -> str:
    """
    使用RoboRefer模型进行图文QA任务（适配benchmark需求）。
    
    Args:
        image: 图像输入，可以是单张或列表（优先支持图像路径列表）。
        question: 用户提出的问题。
        model_path: 模型路径。
        device: 设备。
        torch_dtype: 数据类型。
        max_new_tokens: 最大生成长度。
        temperature: 采样温度。
    
    Returns:
        模型的文本回答。
    """
    # 加载模型（全局单例）
    model, tokenizer, image_processor, conv = load_roborefer_model(model_path, device, torch_dtype)
    
    # 调试信息
    print(f"Debug: Input type: {type(image)}")
    if isinstance(image, np.ndarray):
        print(f"Debug: Image shape: {image.shape}, dtype: {image.dtype}")
    elif isinstance(image, list):
        print(f"Debug: List length: {len(image)}")
        for i, item in enumerate(image[:3]):  # 只打印前3个
            print(f"Debug: Item {i}: type={type(item)}")
            if isinstance(item, str):
                print(f"Debug: Item {i} path: {item}")
            elif isinstance(item, np.ndarray):
                print(f"Debug: Item {i} shape: {item.shape}")
    
    # 加载和展平图像
    pil_images = flatten_image_list(image)
    
    print(f"Debug: Successfully loaded {len(pil_images)} PIL images")
    
    if not pil_images:
        print("Debug: No valid images found in input")
        # 尝试创建一个默认图像作为fallback
        try:
            default_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # 灰色图像
            pil_images = [Image.fromarray(default_image).convert('RGB')]
            print("Debug: Using default gray image as fallback")
        except Exception as e:
            print(f"Debug: Failed to create fallback image: {e}")
            return "Error: No valid images found in input and cannot create fallback"

    # 直接使用问题作为prompt
    prompt = question
    
    try:
        # 使用RoboRefer的generate_content方法进行图文QA
        # 确保prompt_parts只包含PIL图像和字符串
        prompt_parts = []
        for img in pil_images:
            if isinstance(img, Image.Image):
                prompt_parts.append(img)
            else:
                print(f"Warning: Skipping non-PIL image: {type(img)}")
        
        prompt_parts.append(prompt)
        
        print(f"Debug: Using {len(prompt_parts)-1} images and 1 text prompt")
        response = model.generate_content(
            prompt=prompt_parts,
            generation_config=model.default_generation_config
        )
        
        # 清理输出
        response = response.strip() if response else "Sorry, I cannot generate an answer."
        return response
        
    except Exception as e:
        print(f"Debug: generate_content failed: {e}")
        # 回退到手动tokenization方法
        try:
            image_tensor = process_images(pil_images, image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=torch_dtype)

            # 构建对话prompt
            inp = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
            conv_copy = conv.copy()
            conv_copy.append_message(conv_copy.roles[0], inp)
            conv_copy.append_message(conv_copy.roles[1], None)
            full_prompt = conv_copy.get_prompt()

            # Tokenize
            input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(model.device)

            # 设置停止条件
            stop_str = conv_copy.sep if conv_copy.sep_style != clib.SeparatorStyle.TWO else conv_copy.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # 模型推理
            with torch.inference_mode():
                media = {"image": [img for img in image_tensor]}
                media_config = {}
                
                output_ids = model.generate(
                    input_ids,
                    media=media,
                    media_config=media_config,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            # 解码输出
            outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            
            return outputs.strip()
            
        except Exception as e2:
            print(f"Debug: Manual approach also failed: {e2}")
            return f"Error: Failed to answer question - {e2}"

def batch_query_roborefer(image_list: List[Union[str, np.ndarray, Image.Image, List]], 
                         question_list: List[str],
                         model_path: str = "/home/fx/Exp2/test/EmbodiedEval/msjeval/RoboRefer-2B-SFT",
                         device: str = "cuda:0",
                         torch_dtype: torch.dtype = torch.bfloat16) -> List[str]:
    """
    批量使用RoboRefer模型进行推理。
    
    Args:
        image_list: 图像列表。
        question_list: 问题列表。
        model_path: 模型路径。
        device: 设备。
        torch_dtype: 数据类型。
    
    Returns:
        回答列表。
    """
    if len(image_list) != len(question_list):
        raise ValueError("Image list and question list must have the same length")
    
    return [query_roborefer(img, q, model_path, device, torch_dtype) for img, q in zip(image_list, question_list)]

# 兼容性接口（保持向后兼容）
class TestRoboRefer:
    """
    兼容性包装类，保持与现有代码的兼容性。
    """
    
    def __init__(self, 
                 model_path: str = "/home/fx/Exp2/test/EmbodiedEval/msjeval/RoboRefer-2B-SFT",
                 device: str = "cuda:0",
                 torch_dtype: torch.dtype = torch.bfloat16):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        # 预加载模型
        load_roborefer_model(model_path, device, torch_dtype)

    def move_to_device(self, device: str):
        """兼容接口，实际无需操作。"""
        pass

    def chat_image(self, image: Union[str, np.ndarray, Image.Image, List], question: str) -> str:
        """兼容接口。"""
        return query_roborefer(image, question, self.model_path, self.device, self.torch_dtype)

    def generate_image(self, image: Union[str, np.ndarray, Image.Image, List], question: str) -> str:
        """兼容接口。"""
        return self.chat_image(image, question)
    
    def batch_generate(self, image_list: List[Union[str, np.ndarray, Image.Image]], 
                      question_list: List[str]) -> List[str]:
        """兼容接口。"""
        return batch_query_roborefer(image_list, question_list, self.model_path, self.device, self.torch_dtype)

def test_roborefer():
    """测试RoboRefer模型的图文QA功能（benchmark适配）。"""
    print("=== Testing RoboRefer Model for Visual QA ===")
    
    try:
        # 创建简单的测试图像
        test_image_red = np.zeros((224, 224, 3), dtype=np.uint8)
        test_image_red[:, :, 0] = 255  # 红色图像
        
        test_image_blue = np.zeros((224, 224, 3), dtype=np.uint8)  
        test_image_blue[:, :, 2] = 255  # 蓝色图像
        
        print("\n--- Basic QA Test (numpy arrays) ---")
        question = "What color is this image?"
        
        # 测试红色图像
        answer = query_roborefer(test_image_red, question)
        print(f"Red Image - Q: {question} | A: {answer}")
        
        # 测试蓝色图像  
        answer = query_roborefer(test_image_blue, question)
        print(f"Blue Image - Q: {question} | A: {answer}")

        print("\n--- Description Test ---")
        description_question = "Describe what you see in this image."
        answer = query_roborefer(test_image_red, description_question)
        print(f"Description - Q: {description_question} | A: {answer}")

        print("\n--- Deep Nested Numbers Test (simulating pixel data) ---")
        # 模拟深层嵌套的像素数据
        # 创建一个简单的32x32 RGB图像数据
        pixel_data = []
        for i in range(32):
            for j in range(32):
                # 创建渐变效果
                r = int(255 * i / 32)
                g = int(255 * j / 32)
                b = int(255 * (i + j) / 64)
                pixel_data.extend([r, g, b])
        
        # 创建深层嵌套结构
        deep_nested = [[[[pixel_data]]]]  # 4层嵌套
        deep_question = "What do you see in this image?"
        print(f"Testing with deep nested pixel data ({len(pixel_data)} numbers)")
        answer = query_roborefer(deep_nested, deep_question)
        print(f"Deep Nested - Q: {deep_question} | A: {answer}")
        
        print("\n--- Nested List Test (simulating benchmark input) ---")
        # 模拟benchmark中的嵌套列表输入格式
        nested_input = [[test_image_red]]  # 类似 [[image_data]] 的格式
        nested_question = "What color is this image?"
        print(f"Testing with nested input: {type(nested_input)}")
        answer = query_roborefer(nested_input, nested_question)
        print(f"Nested List - Q: {nested_question} | A: {answer}")
        
        # 测试更复杂的嵌套结构
        complex_nested = [[[test_image_blue]], [[test_image_red]]]
        complex_question = "What colors do you see in these images?"
        print(f"Testing with complex nested input")
        answer = query_roborefer(complex_nested, complex_question)
        print(f"Complex Nested - Q: {complex_question} | A: {answer}")
        
        print("\n--- Image Path List Test (benchmark format) ---")
        # 创建临时图像文件来测试路径列表
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        temp_path1 = os.path.join(temp_dir, "test_red.jpg")
        temp_path2 = os.path.join(temp_dir, "test_blue.jpg")
        
        # 保存测试图像
        Image.fromarray(test_image_red).save(temp_path1)
        Image.fromarray(test_image_blue).save(temp_path2)
        
        # 测试图像路径列表（benchmark格式）
        image_paths = [temp_path1, temp_path2]
        path_question = "What colors do you see in these images?"
        print(f"Testing with image paths: {image_paths}")
        answer = query_roborefer(image_paths, path_question)
        print(f"Path List - Q: {path_question} | A: {answer}")
        
        # 清理临时文件
        os.remove(temp_path1)
        os.remove(temp_path2)
        os.rmdir(temp_dir)

        print("\n--- Batch Processing Test ---")
        batch_images = [test_image_red, test_image_blue]
        batch_questions = ["What color is this?", "What color is this?"]
        answers = batch_query_roborefer(batch_images, batch_questions)
        for i, (q, a) in enumerate(zip(batch_questions, answers)):
            print(f"Batch {i+1} - Q: {q} | A: {a}")

        print("\n✅ RoboRefer Visual QA test completed!")

    except Exception as e:
        print(f"\n❌ RoboRefer test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_roborefer() 