from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np

def test_magma_fixed():
    """修复版本的Magma测试，解决感叹号输出问题"""
    
    print("=== Fixed Magma Test ===")
    
    # 加载官方模型
    print("Loading official Magma model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Magma-8B", 
        trust_remote_code=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
    
    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # 正确的prompt格式
    convs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "<image_start><image><image_end>\nWhat color is this image?"},
    ]
    prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    
    print(f"Prompt: {repr(prompt)}")
    
    # 处理输入
    inputs = processor(images=[test_image], texts=prompt, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
    
    # 移动到设备
    inputs = {k: v.to(device=model.device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype) 
              for k, v in inputs.items()}
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # 检查图像token
    input_ids = inputs['input_ids'][0]
    image_token_id = processor.tokenizer.encode("<image>", add_special_tokens=False)[0]
    image_start_id = processor.tokenizer.encode("<image_start>", add_special_tokens=False)[0]
    image_end_id = processor.tokenizer.encode("<image_end>", add_special_tokens=False)[0]
    
    print(f"Image token IDs: <image>={image_token_id}, <image_start>={image_start_id}, <image_end>={image_end_id}")
    print(f"Input contains <image>: {image_token_id in input_ids}")
    print(f"Input contains <image_start>: {image_start_id in input_ids}")
    print(f"Input contains <image_end>: {image_end_id in input_ids}")
    
    # 尝试不同的生成参数
    generation_configs = [
        {
            "name": "Greedy (default)",
            "params": {
                "max_new_tokens": 20,
                "do_sample": False,
                "use_cache": True,
            }
        },
        {
            "name": "Greedy (no temp)",
            "params": {
                "max_new_tokens": 20,
                "do_sample": False,
                "use_cache": True,
                "temperature": None,
            }
        },
        {
            "name": "Sampling (high temp)",
            "params": {
                "max_new_tokens": 20,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
                "use_cache": True,
            }
        },
        {
            "name": "Sampling (low temp)",
            "params": {
                "max_new_tokens": 20,
                "do_sample": True,
                "temperature": 0.1,
                "top_p": 0.9,
                "use_cache": True,
            }
        },
        {
            "name": "Beam search",
            "params": {
                "max_new_tokens": 20,
                "do_sample": False,
                "num_beams": 3,
                "use_cache": True,
            }
        }
    ]
    
    for config in generation_configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        try:
            with torch.inference_mode():
                generate_ids = model.generate(**inputs, **config['params'])
            
            new_tokens = generate_ids[:, inputs["input_ids"].shape[-1]:]
            response = processor.decode(new_tokens[0], skip_special_tokens=True).strip()
            
            print(f"Response: {repr(response)}")
            
            # 检查logits
            if len(new_tokens[0]) > 0:
                print(f"First few token IDs: {new_tokens[0][:5].tolist()}")
                
                # 获取最后一个token的logits
                with torch.inference_mode():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    last_logits = logits[0, -1, :]
                    
                    # 获取top-k tokens
                    top_k = 5
                    top_probs, top_indices = torch.topk(torch.softmax(last_logits, dim=-1), top_k)
                    
                    print(f"Top {top_k} tokens after first generation:")
                    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                        token = processor.tokenizer.decode([idx])
                        print(f"  {i+1}. {repr(token)} (ID: {idx}, prob: {prob:.4f})")
            
            if response and all(c == '!' for c in response.strip()):
                print("⚠️  Still getting exclamation marks")
            elif response:
                print("✓ Got meaningful response!")
                break
            else:
                print("⚠️  Empty response")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # 尝试不同的prompt格式
    print(f"\n--- Testing different prompt formats ---")
    
    alternative_prompts = [
        "What do you see in this image?",
        "Describe the color of this image.",
        "What is the main color in this picture?",
        "Tell me about this image.",
    ]
    
    for alt_prompt in alternative_prompts:
        print(f"\nTesting: {alt_prompt}")
        
        convs = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"<image_start><image><image_end>\n{alt_prompt}"},
        ]
        prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(images=[test_image], texts=prompt, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        inputs = {k: v.to(device=model.device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype) 
                  for k, v in inputs.items()}
        
        try:
            with torch.inference_mode():
                generate_ids = model.generate(
                    **inputs, 
                    max_new_tokens=15,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True
                )
            
            new_tokens = generate_ids[:, inputs["input_ids"].shape[-1]:]
            response = processor.decode(new_tokens[0], skip_special_tokens=True).strip()
            
            print(f"Response: {repr(response)}")
            
            if response and not all(c == '!' for c in response.strip()):
                print("✓ Found working prompt!")
                break
                
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_magma_fixed() 