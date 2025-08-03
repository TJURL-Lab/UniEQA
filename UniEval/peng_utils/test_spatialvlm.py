import os
import sys
import torch
from PIL import Image
import logging

# 添加SpaceMantis路径到Python路径
spacemantis_path = "/home/fx/Exp2/video_model/SpaceMantis"
if spacemantis_path not in sys.path:
    sys.path.insert(0, spacemantis_path)
    print(f"✅ Added {spacemantis_path} to Python path")

# 直接导入SpaceMantis的mllava模块
from models.mllava import MLlavaProcessor, LlavaForConditionalGeneration, chat_mllava

class TestSpatialVLM:
    def __init__(self, device=None):
        # 使用本地SpaceMantis模型
        model_path = "/home/sdd/fx/SpaceMantis/"
        
        print(f"Loading SpatialVLM (SpaceMantis) from: {model_path}")
        
        # 检查模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        try:
            # 设置设备
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                    self.dtype = torch.float16
                else:
                    device = "cpu"
                    self.dtype = torch.float32
            else:
                self.dtype = torch.float16 if 'cuda' in str(device) else torch.float32
            
            self.device = device
            print(f"🎯 Target device: {device}, dtype: {self.dtype}")
            
            # 完全按照原始代码加载
            attn_implementation = None  # or "flash_attention_2"
            
            print("📦 Loading processor...")
            self.processor = MLlavaProcessor.from_pretrained(model_path)
            
            print("🔄 Loading model...")
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                # 多GPU - 使用device_map="auto"
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path, 
                    device_map="auto", 
                    torch_dtype=self.dtype, 
                    attn_implementation=attn_implementation
                )
                self.is_multi_gpu = True
            else:
                # 单GPU - 使用device_map="cuda"
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path, 
                    device_map=device, 
                    torch_dtype=self.dtype, 
                    attn_implementation=attn_implementation
                )
                self.is_multi_gpu = False
            
            print(f"✅ Model loaded successfully!")
            
            # 完全按照原始代码的生成参数
            self.generation_kwargs = {
                "max_new_tokens": 1024,
                "num_beams": 1,
                "do_sample": False
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SpatialVLM: {str(e)}")

    def move_to_device(self, device=None):
        """移动模型到指定设备（兼容model_worker要求）"""
        if device is not None and not self.is_multi_gpu:
            self.device = device
            try:
                if 'cuda' in str(device):
                    self.dtype = torch.float16
                    self.model = self.model.to(device)
                    print(f"✅ SpatialVLM model moved to GPU: {device}")
                else:
                    self.dtype = torch.float32
                    self.model = self.model.to('cpu')
                    print(f"✅ SpatialVLM model moved to CPU")
            except Exception as e:
                print(f"⚠️ Warning: Could not move SpatialVLM model to device {device}: {e}")
        elif self.is_multi_gpu:
            print("ℹ️ Multi-GPU SpatialVLM model already optimally distributed")
        else:
            print(f"ℹ️ SpatialVLM model already on device: {self.device}")

    def run_inference_single(self, image, content):
        """
        按照原始代码的推理方式 - 简化图像类型检查
        """
        try:
            print(f"🤖 Running inference...")
            print(f"📝 Question: {content}")
            print(f"🖼️ Image type: {type(image)}")
            
            # 确保是PIL Image对象
            if hasattr(image, 'size'):
                print(f"🖼️ Image size: {image.size}")
            else:
                raise ValueError(f"Expected PIL Image, got {type(image)}")
            
            # 完全按照原始代码：直接传递图像列表给chat_mllava
            images = [image]
            
            # 使用原始的chat_mllava函数
            response, history = chat_mllava(
                content,           # text
                images,           # images 
                self.model,       # model
                self.processor,   # processor
                **self.generation_kwargs
            )
            
            print(f"📄 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            return response
                
        except Exception as e:
            error_msg = f"SpatialVLM inference failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return error_msg

    def process_multiple_images(self, image_list):
        """处理多张图像，返回结果列表"""
        results = []
        for i, image in enumerate(image_list):
            try:
                print(f"🔄 Processing image {i+1}/{len(image_list)}")
                # 这里可以添加批量处理逻辑，暂时逐个处理
                result = self.run_inference_single(image, "Describe this image.")
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing image {i+1}: {e}")
                results.append(f"Error: {str(e)}")
        return results

    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):
        """
        对图像列表进行推理（EmbodiedEval接口）
        按照test_llava.py的处理方式
        """
        try:
            print(f"🔥 SpatialVLM batch image generation starting...")
            print(f"📊 Processing {len(image_list_list)} image groups")
            
            # 导入contact_img和get_image函数（参考test_llava.py）
            try:
                from . import contact_img, get_image
            except ImportError:
                import sys
                import os
                parent_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, parent_dir)
                import __init__
                contact_img = __init__.contact_img
                get_image = __init__.get_image
            
            # 使用contact_img预处理图像数据（参考test_llava.py）
            contacted_images = contact_img(image_list_list)
            print(f"📷 After contact_img: {len(contacted_images)} images")
            
            results = []
            
            for i, (contacted_image, question) in enumerate(zip(contacted_images, question_list)):
                try:
                    print(f"\n🔄 Processing group {i+1}/{len(contacted_images)}")
                    print(f"📝 Question: {question}")
                    print(f"🖼️ Contacted image type: {type(contacted_image)}")
                    
                    # 使用get_image处理contacted_image（参考test_llava.py）
                    pil_image = get_image(contacted_image)
                    print(f"🖼️ After get_image: {type(pil_image)}")
                    
                    if hasattr(pil_image, 'size'):
                        print(f"🖼️ Final image size: {pil_image.size}")
                    
                    # 运行推理
                    response = self.run_inference_single(pil_image, question)
                    results.append(response)
                    
                except Exception as e:
                    error_msg = f"Error processing group {i+1}: {str(e)}"
                    print(f"❌ {error_msg}")
                    import traceback
                    print(f"🔍 Traceback: {traceback.format_exc()}")
                    results.append(error_msg)
            
            print(f"✅ SpatialVLM batch processing completed!")
            return results
            
        except Exception as e:
            print(f"❌ SpatialVLM batch generation failed: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return [f"Batch processing error: {str(e)}"] * len(image_list_list)

    @torch.no_grad()
    def generate_clip(self, file_list, question_list):
        """
        对视频文件进行推理（EmbodiedEval接口）
        """
        try:
            print(f"🎬 SpatialVLM video generation starting...")
            print(f"📊 Processing {len(file_list)} videos")
            
            results = []
            
            for i, (video_path, question) in enumerate(zip(file_list, question_list)):
                try:
                    print(f"\n🔄 Processing video {i+1}/{len(file_list)}")
                    print(f"🎬 Video: {video_path}")
                    print(f"📝 Question: {question}")
                    
                    # 从视频提取关键帧
                    frames = self.extract_video_frames(video_path)
                    
                    if not frames:
                        print("⚠️ No frames extracted from video")
                        results.append("Could not extract frames from video")
                        continue
                    
                    # 使用第一帧进行推理
                    primary_frame = frames[0]
                    print(f"🖼️ Using first frame from {len(frames)} frames")
                    
                    # 修改问题以适应视频内容
                    video_question = f"This is a frame from a video. {question}"
                    
                    response = self.run_inference_single(primary_frame, video_question)
                    results.append(response)
                    
                except Exception as e:
                    error_msg = f"Error processing video {i+1}: {str(e)}"
                    print(f"❌ {error_msg}")
                    results.append(error_msg)
            
            print(f"✅ SpatialVLM video processing completed!")
            return results
            
        except Exception as e:
            print(f"❌ SpatialVLM video generation failed: {e}")
            return [f"Video processing error: {str(e)}"] * len(file_list)

    def extract_video_frames(self, video_path, num_frames=8):
        """从视频中提取关键帧"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            # 均匀采样帧
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
            
            cap.release()
            print(f"📽️ Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            return [] 