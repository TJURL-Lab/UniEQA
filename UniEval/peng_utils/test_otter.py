import torch
from transformers import CLIPImageProcessor
from .otter.modeling_otter import OtterForConditionalGeneration
from . import get_image, DATA_DIR, contact_img

# CKPT_PATH=f'{DATA_DIR}/otter-9b-hf'
CKPT_PATH="/home/fx/Exp2/test_models/Otter/MODELS/"

class TestOtter:
    def __init__(self, device=None) -> None:
        model_path=CKPT_PATH
        self.model = OtterForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.tokenizer.padding_side = "left"
        self.device = device
        self.model.eval()

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.vision_encoder = self.model.vision_encoder.to(self.device, dtype=self.dtype)
    
    @torch.no_grad()
    def generate_image(self, image_list, question_list, no_image_flag=False):
        image_list = contact_img(image_list)
        imgs = [get_image(img) for img in image_list]
        imgs = [self.image_processor.preprocess([x], return_tensors="pt")["pixel_values"].unsqueeze(0) for x in imgs]
        vision_x = (torch.stack(imgs, dim=0).to(self.device))
        lang_x = [self.tokenizer(
            [
                self.get_formatted_prompt(question, no_image_flag=no_image_flag),
            ],
            return_tensors="pt",
        ) for question in question_list]
        # lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)

        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x["input_ids"].to(self.model.device),
            attention_mask=lang_x["attention_mask"].to(self.model.device),
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output = self.tokenizer.decode(generated_text[0]).split("<answer>")[-1].strip().replace("<|endofchunk|>", "")
        return output

    def get_formatted_prompt(self, question: str, no_image_flag: bool) -> str:
        if no_image_flag:
            return f"User:{question} GPT:<answer>"
        else:
            return f"<image>User:{question} GPT:<answer>"