import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

# 设置常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def split_model(model_path):
    """多GPU模型分布函数 - 参考官方推荐策略，使用GPU 2作为主GPU"""
    device_map = {}
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    if world_size <= 1:
        return None  # 单GPU或CPU，使用默认加载
    
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        print(f"Model has {num_layers} layers")
        
        # 使用GPU 2作为主GPU（承载vision model），将其视为半个GPU
        # 这样GPU 2分配更少的语言模型层，为vision model留出空间
        main_gpu = 2
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu_list = [num_layers_per_gpu] * world_size
        num_layers_per_gpu_list[main_gpu] = math.ceil(num_layers_per_gpu * 0.5)  # 主GPU减半
        
        print(f"Layers per GPU: {num_layers_per_gpu_list} (GPU {main_gpu} is main with fewer layers)")
        
        # 分配语言模型层
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu_list):
            for j in range(num_layer):
                if layer_cnt < num_layers:
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
        
        # 将所有关键组件放在主GPU（GPU 2）
        # 因为我们已经减少了GPU 2的语言模型层数
        device_map['vision_model'] = main_gpu
        device_map['mlp1'] = main_gpu
        device_map['language_model.model.tok_embeddings'] = main_gpu
        device_map['language_model.model.embed_tokens'] = main_gpu
        device_map['language_model.output'] = main_gpu
        device_map['language_model.model.norm'] = main_gpu
        device_map['language_model.model.rotary_emb'] = main_gpu
        device_map['language_model.lm_head'] = main_gpu
        device_map[f'language_model.model.layers.{num_layers - 1}'] = main_gpu
        
        print(f"Device map created for GPUs 0-{world_size-1} (GPU {main_gpu} as vision+embeddings host)")
        print(f"GPU {main_gpu} has fewer language layers to accommodate vision model")
        return device_map
        
    except Exception as e:
        print(f"Failed to create device map: {e}")
        return None

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl(image_input, input_size=448, max_num=12):
    """加载和预处理图像"""
    # 处理不同类型的输入
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype('uint8')).convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class TestInternVL3:
    def __init__(self, device=None):
        # 使用本地模型路径
        model_path = '/home/fx/Exp2/test/EmbodiedEval/msjeval/model_zoo/InternVL3-14B'
        
        print(f"Loading InternVL3 from: {model_path}")
        
        try:
            # 创建多GPU设备映射
            device_map = split_model(model_path)
            
            # 加载模型和tokenizer
            if device_map is not None:
                print("Using multi-GPU device map")
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=False,  # 可以尝试True如果支持
                    trust_remote_code=True,
                    device_map=device_map
                ).eval()
                self.is_multi_gpu = True
            else:
                print("Using single GPU/CPU loading")
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=False,
                    trust_remote_code=True
                ).eval()
                self.is_multi_gpu = False
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                use_fast=False
            )
            
            print("Model and tokenizer loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
        
        self.generation_config = dict(max_new_tokens=512, do_sample=True)
        
        # 初始化dtype和device
        self.dtype = torch.bfloat16
        if self.is_multi_gpu:
            self.device = 'cuda:2'  # 主GPU设为2，承载vision model
        else:
            self.device = 'cpu'
        
        if device is not None and not self.is_multi_gpu:
            self.move_to_device(device)

    def move_to_device(self, device=None):
        # 多GPU情况下跳过设备移动，因为模型已经通过device_map分布
        if self.is_multi_gpu:
            print(f"Skipping device move for multi-GPU setup (model already distributed)")
            return
            
        if device is not None and 'cuda' in str(device):
            self.dtype = torch.bfloat16
            self.device = device
            try:
                self.model.to(device=device, dtype=self.dtype)
                print(f"Model moved to device: {device}")
            except Exception as e:
                print(f"Warning: Could not move model to device {device}: {e}")
        else:
            self.dtype = torch.float32
            self.device = 'cpu'

    @torch.no_grad()
    def generate_image(self, image_list_list, question_list):
        print("InternVL3 image generation begins")
        print(f"Input shape: {np.array(image_list_list).shape}")
        print(f"Questions: {len(question_list)}")
        
        try:
            # 导入contact_img函数
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
            
            # 使用contact_img预处理图像数据（与LLaVA相同的方式）
            contacted_images = contact_img(image_list_list)
            
            answers = []
            for contacted_image, question in zip(contacted_images, question_list):
                try:
                    # 使用get_image处理contacted_image
                    pil_image = get_image(contacted_image)
                    
                    # 加载和预处理图像
                    pixel_values = load_image_internvl(pil_image, max_num=12)
                    
                    # 确保数据类型与模型一致
                    if self.is_multi_gpu:
                        # 多GPU情况下，转换为bfloat16并放到主设备（GPU 2）
                        pixel_values = pixel_values.to(dtype=torch.bfloat16, device='cuda:2')
                    else:
                        # 单GPU情况下，按原来的逻辑处理
                        pixel_values = pixel_values.to(self.dtype)
                        if hasattr(self, 'device') and 'cuda' in str(self.device):
                            pixel_values = pixel_values.to(self.device)
                    
                    # 构建问题
                    formatted_question = f'<image>\n{question}'
                    
                    # 生成回答
                    response = self.model.chat(
                        self.tokenizer, 
                        pixel_values, 
                        formatted_question, 
                        self.generation_config
                    )
                    
                    answers.append(response)
                    print(f"Generated answer: {response[:100]}...")
                    
                except Exception as e:
                    error_msg = f"Error processing image: {str(e)}"
                    print(f"Error in generate_image: {error_msg}")
                    answers.append(error_msg)
            
            print("InternVL3 image generation finished")
            return answers
            
        except Exception as e:
            error_msg = f"Error in generate_image: {str(e)}"
            print(error_msg)
            return [error_msg] * len(question_list)

    @torch.no_grad()
    def generate_clip(self, file_list, question_list):
        print("InternVL3 video generation begins")
        print(f"Videos: {len(file_list)}")
        print(f"Questions: {len(question_list)}")
        
        try:
            # 导入contact_img函数（用于视频处理）
            try:
                from . import contact_img
            except ImportError:
                import sys
                import os
                parent_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, parent_dir)
                import __init__
                contact_img = __init__.contact_img
                
            answers = []
            for video_path, question in zip(file_list, question_list):
                try:
                    # 如果输入是嵌套列表，先用contact_img预处理
                    if isinstance(video_path, list):
                        contacted_path = contact_img([video_path])[0]
                        video_path = contacted_path
                    
                    # 加载和预处理视频
                    pixel_values, num_patches_list = load_video(
                        video_path, 
                        num_segments=8, 
                        max_num=1
                    )
                    
                    # 确保数据类型与模型一致
                    if self.is_multi_gpu:
                        # 多GPU情况下，转换为bfloat16并放到主设备（GPU 2）
                        pixel_values = pixel_values.to(dtype=torch.bfloat16, device='cuda:2')
                    else:
                        # 单GPU情况下，按原来的逻辑处理
                        pixel_values = pixel_values.to(self.dtype)
                        if hasattr(self, 'device') and 'cuda' in str(self.device):
                            pixel_values = pixel_values.to(self.device)
                    
                    # 构建视频前缀和问题
                    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                    formatted_question = video_prefix + question
                    
                    # 生成回答
                    response = self.model.chat(
                        self.tokenizer, 
                        pixel_values, 
                        formatted_question, 
                        self.generation_config,
                        num_patches_list=num_patches_list
                    )
                    
                    answers.append(response)
                    print(f"Generated video answer: {response[:100]}...")
                    
                except Exception as e:
                    error_msg = f"Error processing video: {str(e)}"
                    print(f"Error in generate_clip: {error_msg}")
                    answers.append(error_msg)
            
            print("InternVL3 video generation finished")
            return answers
            
        except Exception as e:
            error_msg = f"Error in generate_clip: {str(e)}"
            print(error_msg)
            return [error_msg] * len(question_list) 