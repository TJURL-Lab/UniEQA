from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor 
from transformers import BitsAndBytesConfig

# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with quantization config
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Magma-8B", 
    trust_remote_code=True,
    device_map={"": 0},  # force everything onto GPU 0
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)

# Inference
image = Image.open("/home/fx/magma_logo.jpg").convert("RGB")

convs = [
    {"role": "system", "content": "You are agent that can see, talk and act."},            
    {"role": "user", "content": "<image_start><image><image_end>\nWhat is the letter on the robot?"},
]
prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
inputs = processor(images=[image], texts=prompt, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)

# Convert inputs to the correct device and data type
inputs = {k: v.to(device=model.device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype) 
          for k, v in inputs.items()}

generation_args = { 
    "max_new_tokens": 500, 
    "temperature": 0.0, 
    "do_sample": False, 
    "use_cache": True,
    "num_beams": 1,
} 

with torch.inference_mode():
    generate_ids = model.generate(**inputs, **generation_args)

generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
print(response)