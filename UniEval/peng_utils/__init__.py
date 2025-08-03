import base64
import gc
import os

import cv2
import torch
import numpy as np
from PIL import Image
from constants import IMAGE_SIZE

DATA_DIR = '/root/VLP_web_data'

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image.astype('uint8')).convert('RGB')
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_BGR_image(image):
    image = get_image(image)
    image = np.array(image)[:, :, ::-1]
    image = Image.fromarray(np.uint8(image))
    return image


def get_model(model_name, device=None):
    if model_name == 'blip2':
        from .test_blip2 import TestBlip2
        return TestBlip2(device)
    elif model_name == 'llava':
        from .test_llava import TestLLaVA
        return TestLLaVA(device)
    elif model_name == 'llava-1.5':
        from .test_llava15 import TestLLaVA15
        return TestLLaVA15(device)
    elif model_name == 'videollava':
        from .test_videollava import TestVideoLLaVA
        return TestVideoLLaVA(device)
    elif model_name == 'minicpm':
        from .test_minicpm import TestMiniCPM
        return TestMiniCPM(device)
    elif model_name == 'instructblip':
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP(device)
    elif model_name == 'gpt-4o':
        from .test_gpt4o import TestGPT4O
        return TestGPT4O(device)
    elif model_name == 'Seed-1.5-VL':
        from .test_seed15vl import TestSEED15VL
        return TestSEED15VL(device)
    elif model_name == 'minigpt4':
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(device)
    elif model_name == 'llama-vid-image':
        from .test_llama_vid_image import TestLLaMAVID
        return TestLLaMAVID(device)
    elif model_name == 'llama-vid-video':
        from .test_llama_vid_video import TestLLaMAVID
        return TestLLaMAVID(device)
    elif model_name == 'pllava':
        from .test_pllava import TestPLLaVA
        return TestPLLaVA(device)
    elif model_name == 'llava-next-video':
        from .test_llava_next_video import TestLLaVANextVideo
        return TestLLaVANextVideo(device)
    elif model_name == 'llava-next-video_1':
        from .test_llava_next_video import TestLLaVANextVideo
        return TestLLaVANextVideo(device)
    elif model_name == 'MiniCPM-V 2.6':
        from .test_minicpm_v_2_6 import TestMiniCPM_V_2_6
        return TestMiniCPM_V_2_6(device)
    elif model_name == 'EmbodiedGPT':
        from .test_embodiedgpt import TestEmbodiedGPT
        return TestEmbodiedGPT(device)
    elif model_name == 'Fine-tuned MiniCPM-V':
        from .test_tuned_minicpm_v_2_6 import TestTunedMiniCPM_V_2_6
        return TestTunedMiniCPM_V_2_6(device)
    elif model_name == 'minicpm3-4b':
        from .test_minicpm_3_4b import TestMiniCPM_3_4b
        return TestMiniCPM_3_4b(device)
    elif model_name == 'Qwen2.5-VL':
        from .test_qwen2_5 import Testqwen2_5
        return Testqwen2_5(device)
    elif model_name == 'Qwen2.5-VL-7B-Ins':
        from .test_qwen2_5_7b_optimized import Testqwen2_5_7b_optimized
        return Testqwen2_5_7b_optimized(device)
    elif model_name == 'QwenVLMax':
        from .test_qwenvlmax import TestQwenVLMax
        return TestQwenVLMax(device)
    elif model_name == 'vebrain':
        from .test_vebrain import Testvebrain
        return Testvebrain(device)
    elif model_name == 'phi3':  
        from .test_phi3 import TestPhi3
        return TestPhi3(device)
    elif model_name == 'minicpmv':
        from .test_minicpmv import TestMiniCPMV
        return TestMiniCPMV(device)
    elif model_name == 'gemini-2.5-pro':
        from .test_gemini25pro import TestGemini25Pro
        return TestGemini25Pro(device)
    # elif model_name == 'pllava_1':
    #     from .test_pllava import TestPLLaVA
    #     return TestPLLaVA(device)
    # elif model_name == 'pllava_2':
    #     from .test_pllava import TestPLLaVA
    #     return TestPLLaVA(device)
    # elif model_name == 'pllava_3':
    #     from .test_pllava import TestPLLaVA
    #     return TestPLLaVA(device)

    elif model_name == 'llava-onevision':
        from .test_llava_onevision import TestLLaVAOneVision
        return TestLLaVAOneVision(device)
    elif model_name == 'llava-onevision-0.5b':
        from .test_llava_onevision_05b import TestLLaVAOneVision
        return TestLLaVAOneVision(device)
    elif model_name == 'internvl3':
        from .test_internvl3 import TestInternVL3
        return TestInternVL3(device)
    elif model_name == 'internvl3-14b':
        from .test_internvl3_14b import TestInternVL3
        return TestInternVL3(device)
    elif model_name == 'owl':
        from .test_mplug_owl import TestMplugOwl
        return TestMplugOwl(device)
    elif model_name == 'RoboBrain':
        from .test_robobrain import TestRoboBrain
        return TestRoboBrain(device)
    elif model_name == 'Cosmos-R1':
        from .test_cosmos_r1 import TestCosmosR1
        return TestCosmosR1(device)
    elif model_name == 'RoboPoint':
        from .test_robopoint import TestRoboPoint
        return TestRoboPoint(device)
    elif model_name == 'otter':
        from .test_otter import TestOtter
        return TestOtter(device)
    elif model_name == 'vpgtrans':
        from .test_vpgtrans import TestVPGTrans
        return TestVPGTrans(device)
    elif model_name == 'llama_adapter_v2':
        from .test_llama_adapter_v2 import TestLLamaAdapterV2, TestLLamaAdapterV2_web
        return TestLLamaAdapterV2(device)
    elif model_name == 'o3':
        from .test_o3 import TestO3
        return TestO3(device)
    elif model_name == 'VILA' or model_name == 'VILA1.5-13B':
        from .test_vila import TestVILA
        return TestVILA(device)
    elif model_name == 'spatialvlm' or model_name == 'SpatialVLM' or model_name == 'SpaceMantis':
        from .test_spatialvlm import TestSpatialVLM
        return TestSpatialVLM(device)
    elif model_name == 'magma' or model_name == 'Magma' or model_name == 'Magma-8B':
        from .test_magma import TestMagma
        return TestMagma(device)
    elif model_name == 'RoboRefer':
        from .test_roborefer import TestRoboRefer
        return TestRoboRefer(device=str(device) if device else "cuda:0")
    elif model_name == 'SAT' or model_name == 'sat':
        from .test_sat import TestSAT
        return TestSAT(device)
    elif model_name == 'embodied-r1':
        from .test_embodied_r1 import TestEmbodiedR1
        return TestEmbodiedR1(device)
    # elif model_name == 'instruct_blip':
    #     from .test_instructblip import TestInstructBLIP
    #     return TestInstructBLIP(device)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")


def get_device_name(device: torch.device):
    return f"{device.type}{'' if device.index is None else ':' + str(device.index)}"

def process_image(img_list):
    images_list = []
    flag = True
    for p in img_list:
        if flag:
            p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
            images_list.append(p)
            width = p.shape[1]
            height = p.shape[0]
            img_size = (width, height)
            flag = False
        else:
            p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
            p = cv2.resize(p, img_size, 0, 0, cv2.INTER_LINEAR)
            images_list.append(p)
    im_v = cv2.vconcat(images_list)

    # retval, buffer = cv2.imencode('.jpg', im_v)
    # if not retval:
    #     print("Could not encode image.")
    #     return None
    #
    # im_64 = base64.b64encode(buffer)
    return im_v

def contact_img(array_list_list):
    contacted_image_list = []
    for img_list in array_list_list:
        image_list = []
        for img in img_list:
            img = np.array(img, dtype='uint8')
            # img = Image.fromarray(img.astype('uint8')).convert('RGB')   # .resize((IMAGE_SIZE, IMAGE_SIZE))
            image_list.append(img)
        contacted_image_list.append(process_image(image_list))
    print(f"contacted_image_list: {len(contacted_image_list)}")  # \n{image_list_list}")
    return contacted_image_list

@torch.inference_mode()
def generate_stream(model, text, visual_info, data_type, device=None, keep_in_device=False):
    print("generate_stream begins")

    if device != model.device:
        model.move_to_device(device)

    if data_type=='image':
        print("image generate")
        output = model.generate_image(visual_info, text)
    elif data_type=='video':
        print("video generate")
        output = model.generate_clip(visual_info, text)
    else:
        print('unsupported data type')
        output = 'unsupported data type'
    # else:
    #     print("single image")
    #     # print(f"before np.array:{image}")
    #     image = np.array(image, dtype='uint8')
    #     # print(f"after np.array:{image.shape}")
    #     image = Image.fromarray(image.astype('uint8')).convert('RGB')

    # output = model.generate(image, text, device, keep_in_device)

    if not keep_in_device:
        model.move_to_device(None)
    print(f"{'#' * 20} Model out: {output}")
    gc.collect()
    torch.cuda.empty_cache()
    print("generate_stream finish")
    yield output
