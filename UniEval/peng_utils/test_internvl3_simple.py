#!/usr/bin/env python3

import torch
import numpy as np
from PIL import Image
import sys
import os

def test_internvl3():
    print("Testing InternVL3 model...")
    
    try:
        # 直接导入模型类
        from test_internvl3 import TestInternVL3
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"Using CUDA device: {device}")
        else:
            device = torch.device('cpu')
            print("Using CPU device")
        
        # 初始化模型
        print("Initializing InternVL3 model...")
        model = TestInternVL3(device=device)
        print("Model loaded successfully!")
        
        # 创建测试图像数据
        print("Creating test data...")
        # 模拟EmbodiedEval的数据格式：嵌套list结构
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_list_list = [[test_image]]  # 嵌套list结构
        question_list = ["What do you see in this image?"]
        
        print(f"Test data shape: {np.array(image_list_list).shape}")
        
        # 测试图像生成
        print("Testing image generation...")
        answers = model.generate_image(image_list_list, question_list)
        print(f"Generated answers: {answers}")
        
        print("InternVL3 test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_internvl3()
    sys.exit(0 if success else 1) 