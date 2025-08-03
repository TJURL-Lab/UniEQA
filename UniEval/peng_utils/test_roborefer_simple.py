import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
from typing import List, Union, Optional
import llava
from llava import conversation as clib
from llava.media import Image as LLaVAImage, Video, Depth

# 添加RoboRefer路径
sys.path.append('/home/fx/Exp2/test/EmbodiedEval/RoboRefer')

class TestRoboRefer:
    """RoboRefer模型推理类，支持多卡推理和class list数据处理"""
    
    def __init__(self, 
                 model_path: str = "/home/fx/Exp2/test/EmbodiedEval/msjeval/RoboRefer-2B-SFT",
                 device_map: str = "auto",
                 torch_dtype: torch.dtype = torch.bfloat16):
        """
        初始化RoboRefer模型
        
        Args:
            model_path: RoboRefer模型路径
            device_map: 设备映射策略 ("auto", "balanced", "sequential" 或具体映射)
            torch_dtype: 模型数据类型
        """
        self.torch_dtype = torch_dtype
        
        print(f"Loading RoboRefer model from {model_path}")
        print(f"Device map: {device_map}")
        print(f"Torch dtype: {torch_dtype}")
        
        # 初始化VLM模型
        self._init_vlm_model(model_path, device_map)
        
        print("RoboRefer model loaded successfully!")
    
    def _init_vlm_model(self, model_path: str, device_map: str):
        """初始化VLM模型"""
        print("Initializing VLM model...")
        
        vlm_conv_mode = 'auto'
        
        try:
            # 使用device_map进行多卡分配
            self.vlm_model = llava.load(
                model_path,
                device_map=device_map,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True
            )
            clib.default_conversation = clib.conv_templates[vlm_conv_mode].copy()
            print("VLM model loaded successfully!")
        except Exception as e:
            print(f"Error loading VLM model: {e}")
            raise
    
    def move_to_device(self, device: str):
        """移动模型到指定设备（兼容peng_utils接口）"""
        print(f"Model device updated to {device}")
        # 注意：使用device_map时，模型会自动管理设备分配
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """加载图像为numpy数组"""
        try:
            if isinstance(image_input, str):
                # 文件路径
                if os.path.exists(image_input):
                    return cv2.imread(image_input)
                else:
                    raise FileNotFoundError(f"Image file not found: {image_input}")
            elif isinstance(image_input, np.ndarray):
                # 已经是numpy数组
                return image_input
            elif isinstance(image_input, Image.Image):
                # PIL图像
                return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
    
    def _save_temp_image(self, image: np.ndarray, prefix: str = "temp") -> str:
        """保存临时图像文件"""
        import tempfile
        import uuid
        
        temp_dir = tempfile.gettempdir()
        filename = f"{prefix}_{uuid.uuid4().hex}.png"
        filepath = os.path.join(temp_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath
    
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
    
    def _chat_single_image(self, 
                          image: Union[str, np.ndarray, Image.Image], 
                          question: str) -> str:
        """处理单个图像的对话"""
        # 加载图像
        image_array = self._load_image(image)
        
        # 保存临时图像文件
        image_path = self._save_temp_image(image_array, "image")
        
        try:
            # 构建prompt
            prompt = []
            
            # 添加图像
            if image_path.endswith(('.jpg', '.jpeg', '.png')):
                prompt.append(LLaVAImage(image_path))
            elif image_path.endswith(('.mp4', '.mkv', '.webm')):
                prompt.append(Video(image_path))
            else:
                raise ValueError(f"Unsupported media type: {image_path}")
            
            # 添加文本
            if question:
                prompt.append(question)
            
            # 生成回答
            answer = self.vlm_model.generate_content(prompt)
            
            return answer
            
        finally:
            # 清理临时图像文件
            try:
                os.remove(image_path)
            except:
                pass
    
    def _chat_multiple_images(self, 
                             images: List[Union[str, np.ndarray, Image.Image]], 
                             question: str) -> str:
        """处理多个图像的对话"""
        # 构建prompt
        prompt = []
        temp_files = []
        
        try:
            for i, image in enumerate(images):
                # 加载图像
                image_array = self._load_image(image)
                
                # 保存临时图像文件
                image_path = self._save_temp_image(image_array, f"image_{i}")
                temp_files.append(image_path)
                
                # 添加图像到prompt
                if image_path.endswith(('.jpg', '.jpeg', '.png')):
                    prompt.append(LLaVAImage(image_path))
                elif image_path.endswith(('.mp4', '.mkv', '.webm')):
                    prompt.append(Video(image_path))
                else:
                    raise ValueError(f"Unsupported media type: {image_path}")
            
            # 添加文本
            if question:
                prompt.append(question)
            
            # 生成回答
            answer = self.vlm_model.generate_content(prompt)
            
            return answer
            
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def generate_image(self, image: Union[str, np.ndarray, Image.Image, List], question: str) -> str:
        """
        生成图像回答（兼容peng_utils接口）
        
        Args:
            image: 图像输入
            question: 问题文本
        
        Returns:
            模型回答
        """
        return self.chat_image(image, question)
    
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
                answer = self.chat_image(image, question)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                answers.append(f"Error: {str(e)}")
        
        return answers


def test_roborefer():
    """测试RoboRefer模型"""
    print("=== Testing RoboRefer Model ===")
    
    # 初始化模型（使用多卡）
    try:
        model = TestRoboRefer(
            model_path="/home/fx/Exp2/test/EmbodiedEval/msjeval/RoboRefer-2B-SFT",
            device_map="auto",  # 自动多卡分配
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    # 创建测试图像
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # 灰色图像
    
    # 测试单个图像
    print("\n--- Single Image Test ---")
    question = "What color is this image?"
    print(f"Question: {question}")
    
    try:
        answer = model.chat_image(test_image, question)
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
        answer = model.chat_image(test_images, question)
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
    test_roborefer() 