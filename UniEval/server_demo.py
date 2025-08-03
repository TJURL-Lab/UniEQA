"""
The gradio demo server for chatting with a single model.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import PIL
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import re
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import time
import json
import uuid
import random
import datetime
import argparse
import requests
import multiprocessing

import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from server_utils.constants import (
    LOGDIR, WORKER_API_TIMEOUT, CONVERSATION_SAVE_DIR,
    rules_markdown, notice_markdown, license_markdown
)
from server_utils.utils import build_logger, server_error_msg



os.makedirs(CONVERSATION_SAVE_DIR, exist_ok=True)
os.makedirs(f"{CONVERSATION_SAVE_DIR}/images", exist_ok=True)
logger = build_logger("web_server", f"{LOGDIR}/web_server.log")
headers = {"User-Agent": "fastchat Client"}
model_list = []
controller_url = None
# no_change_btn = gr.Button.update()
# enable_btn = gr.Button.update(interactive=True)
# disable_btn = gr.Button.update(interactive=False)

no_change_btn = gr.update()
enable_btn = gr.update(interactive=True)
disable_btn = gr.update(interactive=False)
# ["llava", "llava-1.5", "blip2", "minigpt4", "instructblip", "minicpm", "videollava", "video-chatgpt", "llama-vid", "gpt-4v"]
model_info = {
    'blip2': 'BLIP2-flan-t5-xl-3B',
    'instructblip': 'Instructblipflant5xl',
    'minigpt4': 'MiniGPT-4-7B',
    'minicpm': 'MiniCPM-Llama3-V-2_5',
    'gpt-4v' : 'gpt-4-vision-preview',
    'llava-1.5': 'llava-v1.5-7b',
    'llava': 'LLava-7B',

    'videollava': 'Video-LLaVA-7B',
    'video-chatgpt': '',
    'llama-vid': 'llama-vid-7b-full-336'
    # 'flamingo': 'OpenFlamingo-9B',
    # 'owl': 'mPLUG-Owl-Pretrained-7B',
    # 'otter': 'Otter-9B',
    # 'instruct_blip': 'InstructBLIP-7B',
    # 'vpgtrans': 'VPGTrans-7B',
    # 'llama_adapter_v2': 'LLama-Adapter-V2-7B',
}

bench_info = {
    "img_bench_list": ["EgoThink", "MFE-ETP", "OpenEQA", "ALFRED", "EgoPlan-Bench", "test"],
    "video_bench_list":  ["EgoPlan-Bench", "EgoTaskQA"],
    "test": ["task_completion_condition"],
    "EgoThink": ["action_perception", "affordance", "object_property", "object_type", "open-loop_planning", "situated_reasoning", "spatial_perception", "world_knowledge", "temporal_perception"],
    "MFE-ETP":  ["affordance", "object_property", "object_type", "spatial_perception", "temporal_perception", "closed-loop_planning", "task_completion_condition", "task_related_object"],  # "open-loop_planning",
    "OpenEQA":  ["object_state", "object_property", "object_type", "affordance", "spatial_perception", "world_knowledge"],
    "EgoPlan-Bench":  ["closed-loop_planning"],
    "ALFRED":  ["closed-loop_planning"],
    "EgoTaskQA":  ["object_type", "object_property", "temporal_perception", "situated_reasoning", "object_state", "spatial_perception", "action_perception"]
}

OPENEQA_NUM_FRAMES = 15



video_models = ["videollava", "video-chatgpt", "llama-vid", "videollama", "p-llava"]


class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                    re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip().lower()
        return answer

    def evaluate_rouge(self, preds):
        rouge = Rouge()
        acc = {'f': []}
        eval_list = []
        for i, res in enumerate(preds):
            sample_id = res['sample_id']
            gt_ans = self.process(res["gt_response"])
            pred_ans = self.process(res["pred_response"])
            assert gt_ans != ''
            if pred_ans == '':
                s = 0
            else:
                s = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(s)
            eval_list.append({'id': str(sample_id), 'score': str(round(s, 3))})
        results = {'Rouge-L f': np.mean(acc['f'])}
        return results, eval_list

    def get_choice_list(self, preditions, core_json):
        assert len(preditions) == len(core_json['data'])
        new_pres = {d['sample_id']: d for d in preditions}
        for sample in core_json['data']:
            choice_list = sample['task_instance']['choice_list']
            new_pres[int(sample['sample_id'])]['choice_list'] = choice_list
            new_pres[int(sample['sample_id'])]['gt_response'] = sample['response']
        for pre in new_pres.values():
            assert 'choice_list' in pre.keys()

    def judge_multi_choice(self, sample):
        sample_id = sample['sample_id']
        gt_ans = sample["gt_response"]
        pred_ans = sample["pred_response"]
        choice_list = sample['choice_list']
        if gt_ans not in choice_list:
            print(gt_ans)
            print(choice_list)
        assert gt_ans in choice_list
        try:
            vectorizer = TfidfVectorizer()
            texts = [pred_ans] + choice_list
            tfidf_matrix = vectorizer.fit_transform(texts)
            # print(tfidf_matrix)
            # print(vectorizer.get_feature_names())
            # print(tfidf_matrix.toarray())
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            most_similar_index = cosine_similarities.argmax()
            if choice_list[most_similar_index] == gt_ans:
                return 1
            else:
                return 0
        except:
            if pred_ans == gt_ans:
                return 1
            else:
                return 0

    def process_sample(self, sample):
        sample["gt_response"] = self.process(sample["gt_response"])
        sample["pred_response"] = self.process(sample["pred_response"])
        for i in range(len(sample['choice_list'])):
            sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multichoice(self, preds, core_json):
        self.get_choice_list(preds, core_json)
        correct = 0
        eval_list = []
        for i, sample in enumerate(preds):
            if 'choice_list' not in sample.keys():
                print(sample)
            self.process_sample(sample)
            score = self.judge_multi_choice(sample)
            sample_id = sample['sample_id']
            sample['result'] = score
            eval_list.append({'id': str(sample_id), 'score': str(score)})
            correct += score
        return {'Accuracy': correct / len(preds)}, eval_list

    def judge_multi_answer(self, sample):
        sample_id = sample['sample_id']
        gt_ans_list = sample["gt_response"]  # gt_response是一个列表，包含多个正确答案
        pred_ans_list = sample["pred_response"]  # pred_response是一个列表，包含多个预测答案
        choice_list = sample['choice_list']

        # 检查所有的正确答案是否都在选项列表中
        for gt_ans in gt_ans_list:
            if gt_ans not in choice_list:
                print(f"Ground truth answer '{gt_ans}' not in choice list.")
                print(f"Choice list: {choice_list}")
            assert gt_ans in choice_list, "Ground truth answer is not in the choice list."

        try:
            # 初始化TF-IDF向量化器
            vectorizer = TfidfVectorizer()

            # 将预测答案与选项列表合并，进行TF-IDF计算
            all_texts = pred_ans_list + choice_list
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # 初始化得分
            score = 0

            # 对每个预测答案计算与所有选项的相似度
            for i, pred_ans in enumerate(pred_ans_list):
                cosine_similarities = cosine_similarity(tfidf_matrix[i:i + 1],
                                                        tfidf_matrix[len(pred_ans_list):]).flatten()

                # 选出与当前预测答案最相似的选项
                most_similar_index = cosine_similarities.argmax()
                most_similar_choice = choice_list[most_similar_index]

                # 如果最相似的选项是正确答案之一，则得分+1
                if most_similar_choice in gt_ans_list:
                    score += 1

            # 返回总得分（可以是准确率或其他评价指标）
            return score / len(pred_ans_list)  # 返回平均准确率
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # 简单的相等比较作为回退机制
            score = sum(1 for pred_ans in pred_ans_list if pred_ans in gt_ans_list)
            return score / len(pred_ans_list)  # 返回平均准确率

    def process_sample_multi_answer(self, sample):
        sample["gt_response"] = self.process(sample["gt_response"])
        sample["pred_response"] = self.process(sample["pred_response"])
        for i in range(len(sample['choice_list'])):
            sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multi_answer(self, preds, core_json):
        self.get_choice_list(preds, core_json)
        correct = 0
        eval_list = []
        for i, sample in enumerate(preds):
            if 'choice_list' not in sample.keys():
                print(sample)
            self.process_sample_multi_answer(sample)
            score = self.judge_multi_choice(sample)
            sample_id = sample['sample_id']
            sample['result'] = score
            eval_list.append({'id': str(sample_id), 'score': str(score)})
            correct += score
        return {'Accuracy': correct / len(preds)}, eval_list

def get_model_list(controller_url):
    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(controller_url + "/list_models")
    return list(set(ret.json()["models"]))


def get_model_worker_addr(model_name):
    print("************ get_model_worker_addr ************")
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    print(f"************ get_model_worker_addr finished: {ret.json()['address']}************")
    return ret.json()["address"]


def save_vote_data(state, request: gr.Request):
    if state['data_type']=='image':
        t = datetime.datetime.now()
        # save image
        for img in state['visual']:
            img_name = os.path.join(CONVERSATION_SAVE_DIR, 'images', f"{t.year}-{t.month:02d}-{t.day:02d}-{str(uuid.uuid4())}.png")
            while os.path.exists(img_name):
                img_name = os.path.join(CONVERSATION_SAVE_DIR, 'images', f"{t.year}-{t.month:02d}-{t.day:02d}-{str(uuid.uuid4())}.png")

            image = np.array(img, dtype='uint8')
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
            image.save(img_name)
        # save conversation
        log_name = os.path.join(CONVERSATION_SAVE_DIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conversation.json")
        with open(log_name, 'a') as fout:
            log_data = state.copy()
            log_data['image'] = img_name
            log_data['ip'] = request.client.host
            fout.write(json.dumps(state) + "\n")
    elif state['data_type']=='video':
        t = datetime.datetime.now()
        log_name = os.path.join(CONVERSATION_SAVE_DIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conversation.json")
        with open(log_name, 'a') as fout:
            log_data = state.copy()
            # log_data['video'] = state['visiual']
            log_data['ip'] = request.client.host
            fout.write(json.dumps(state) + "\n")
    else:
        print("unsupported data type for 'save_vote_data'")


def vote_left_model(state, model_name_A, model_name_B, request: gr.Request):
    state['user_vote'] = 'left'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    model_name_B = f"""**Model B: {model_info[state['VLP_names'][1]]}**"""
    save_vote_data(state, request)
    return model_name_A, model_name_B, disable_btn, disable_btn, disable_btn, disable_btn


def vote_right_model(state, model_name_A, model_name_B, request: gr.Request):
    state['user_vote'] = 'right'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    model_name_B = f"""**Model B: {model_info[state['VLP_names'][1]]}**"""
    save_vote_data(state, request)
    return model_name_A, model_name_B, disable_btn, disable_btn, disable_btn, disable_btn


def vote_model_tie(state, model_name_A, model_name_B, request: gr.Request):
    state['user_vote'] = 'tie'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    model_name_B = f"""**Model B: {model_info[state['VLP_names'][1]]}**"""
    save_vote_data(state, request)
    return model_name_A, model_name_B, disable_btn, disable_btn, disable_btn, disable_btn


def vote_model_bad(state, model_name_A, model_name_B, request: gr.Request):
    state['user_vote'] = 'bad'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    model_name_B = f"""**Model B: {model_info[state['VLP_names'][1]]}**"""
    save_vote_data(state, request)
    return model_name_A, model_name_B, disable_btn, disable_btn, disable_btn, disable_btn

def vote_model_good_single(state, model_name_A, request: gr.Request):
    state['user_vote'] = 'bad'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    save_vote_data(state, request)
    return model_name_A, disable_btn, disable_btn

def vote_model_bad_single(state, model_name_A, request: gr.Request):
    state['user_vote'] = 'bad'
    model_name_A = f"""**Model A: {model_info[state['VLP_names'][0]]}**"""
    save_vote_data(state, request)
    return model_name_A, disable_btn, disable_btn


def clear_chat(state):
    if state is not None:
        if "data_type" in state.keys():
            tem_type = state["data_type"]
            state = {'data_type':tem_type}
        else:
            state = {}
    return state, None, None, gr.update(value=None), gr.update(value=None), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, enable_btn


def clear_chat_single(state):
    if state is not None:
        state = {'data_type':'image'}
    return state, None, gr.update(value=None), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), disable_btn, disable_btn, disable_btn, disable_btn

def share_click(state):
    return state


def user_ask(state, chatbot_A, chatbot_B, textbox, imagebox):
    if (textbox == '') or (imagebox is None and 'image' not in state):
    # print(f"textbox:{textbox} type:{type(textbox)}")
    # if (textbox is None) or (imagebox is None and 'image' not in state):
        state['get_input'] = False
        return state, chatbot_A, chatbot_B, "", "", textbox, imagebox, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn
    if imagebox is not None:
        state['image'] = np.array(imagebox, dtype='uint8').tolist()
        imagebox.save('.tmp_img.png')
    print("************ ask ************")
    state['text'] = textbox
    state['get_input'] = True
    selected_VLP_models = random.sample(model_list, 2)
    state['VLP_names'] = selected_VLP_models
    chatbot_A = chatbot_A + [(('.tmp_img.png',), None), (textbox, None), (None, None)]
    chatbot_B = chatbot_B + [(('.tmp_img.png',), None), (textbox, None), (None, None)]
    print("************ ask finished ************")
    return state, chatbot_A, chatbot_B, gr.update(value=None), gr.update(value=None), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn


def get_image_array(images_path_list):
    image_array_list = []
    path_list = []
    for i, path in enumerate(images_path_list):
        img = Image.open(path)
        image_array_list.append(np.array(img, dtype='uint8').tolist())
        path = ".tmp_img" + str(i) + ".png"
        img.save(path)
        path_list.append(path)
    return image_array_list, path_list

def user_ask_single(state, chatbot_A, textbox, radio, radio_video=None):
    print(f"usr asking: type:{state['data_type']}")
    visual = textbox['files']
    textbox = textbox['text']
    if state['data_type']=="image":
        if (textbox == '') or (visual is None and 'visual' not in state):
        # print(f"textbox:{textbox} type:{type(textbox)}")
        # if (textbox is None) or (imagebox is None and 'image' not in state):
            state['get_input'] = False
            return state, chatbot_A, "", textbox, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn
        if visual is not None:
            img_list, path_list = get_image_array(visual)
            state['visual'] = [img_list]    # [[img_array1, img_array2]]
            # visual.save('.tmp_img.png') # possible wrong
        print("************ ask ************")
        state['text'] = [textbox]           #  ['What am I doing?']
        # print(f"********************** state['visual']: {state['visual'][0][0][0]} ****************************")
        # print(f"********************** state['text']: {state['text']} ****************************")
        state['get_input'] = True
        state['VLP_names'] = [radio]
        print(f"path_list: {path_list}\n{tuple(path_list)}")
        print(f"********************** VLP names: {radio}")
        chatbot_A = chatbot_A + [((image,), None) for image in path_list] + [(textbox, None), (None, None)]
        print(f"************ ask finished ************")
        return state, chatbot_A, gr.update(value=None), gr.update(value=None, interactive=True), disable_btn, disable_btn, disable_btn, disable_btn, disable_btn

    elif state['data_type']=="video":
        if (textbox == '') or (visual is None and 'visual' not in state):
        # print(f"textbox:{textbox} type:{type(textbox)}")
        # if (textbox is None) or (imagebox is None and 'image' not in state):
            state['get_input'] = False
            return state, chatbot_A, "", textbox, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn
        if visual is not None:
            state['visual'] = visual    # ['xxxxxx/xxxxxxx.mp4']
        print("************ ask ************")
        state['text'] = textbox  # "What is in the video?"
        # print(f"********************** state['visual']: {state['visual']} ****************************")
        # print(f"********************** state['text']: {state['text']} ****************************")
        state['get_input'] = True
        state['VLP_names'] = [radio_video]
        print(f"********************** VLP names: {radio_video}")
        chatbot_A = chatbot_A + [((None), None), (textbox, None), (None, None)]
        print("************ ask finished ************")
        return state, chatbot_A, gr.update(value=None), gr.update(value=None, interactive=True), disable_btn, disable_btn, disable_btn, disable_btn, disable_btn


def model_worker_stream_iter(worker_addr, state):
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json={"text": state['text'], "visual": state['visual'], "data_type": state['data_type']},
        stream=True,
        timeout=WORKER_API_TIMEOUT,# WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):    # read per line
        if chunk:   # if data is not None: decode to str
            data = json.loads(chunk.decode())
            yield data


def get_model_worker_output(worker_input):
    print("************ get_model_worker_output ************")
    (worker_addr, state) = worker_input
    stream_iter = model_worker_stream_iter(worker_addr, state)
    try:
        for data in stream_iter:
            if data["error_code"] == 0:
                if isinstance(data["text"], list):
                    for i, s in enumerate(data["text"]):
                        # print(data["text"])
                        # print(i, s)
                        data["text"][i] = s.strip()
                    output = data["text"]
                else:
                    output = data["text"].strip()   # delete space
                print(f"************ get_model_worker_output finished: {data}************")
                return output
            elif data["error_code"] == 1:
                output = data["text"] + f" (error_code: {data['error_code']})"
                print(f"************ get_model_worker_output finished: {data}************")
                return output
            time.sleep(0.02)
    except requests.exceptions.RequestException as e:
        output = server_error_msg + f" (error_code: 4)"
        print(f"************ get_model_worker_output finished: {data}************")
        return output
    except Exception as e:
        output = server_error_msg + f" (error_code: 5, {e})"
        print(f"************ get_model_worker_output finished: {data}************")
        return output


def run_VLP_models(state, chatbot_A, chatbot_B):
    if 'get_input' not in state or not state['get_input']:
        return state, chatbot_A, chatbot_B, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, disable_btn, enable_btn
    print("************ run_VLP_models ************")
    model_worker_addrs = [get_model_worker_addr(model_name) for model_name in state['VLP_names']]
    print(f"model_worker_addrs:{model_worker_addrs}")
    pool = multiprocessing.Pool()
    vlp_outputs = pool.map(get_model_worker_output, [(worker_addr, state) for worker_addr in model_worker_addrs])
    state['VLP_outputs'] = vlp_outputs
    chatbot_A[-1][1] = vlp_outputs[0]
    chatbot_B[-1][1] = vlp_outputs[1]
    print(f"************ run_VLP_models finished, A output:{vlp_outputs[0]} B output:{vlp_outputs[1]}************")
    return state, chatbot_A, chatbot_B, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn


def run_VLP_models_single(state, chatbot_A):
    if 'get_input' not in state or not state['get_input']:
        return state, chatbot_A, disable_btn, disable_btn, disable_btn, disable_btn, enable_btn
    print("************ run_VLP_models ************")
    model_worker_addrs = [get_model_worker_addr(model_name) for model_name in state['VLP_names']]
    print(f"model_worker_addrs:{model_worker_addrs}")
    # pool = multiprocessing.Pool()
    # vlp_outputs = pool.map(get_model_worker_output, [(worker_addr, state) for worker_addr in model_worker_addrs])
    vlp_outputs = get_model_worker_output((model_worker_addrs[0], state))
    if isinstance(vlp_outputs, list):
        vlp_outputs = vlp_outputs[0]
    state['VLP_outputs'] = vlp_outputs
    chatbot_A[-1][1] = vlp_outputs
    print(f"************ run_VLP_models_single finished, A output:{vlp_outputs}************")
    return state, chatbot_A, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn

def change_name(name):
    return gr.update(label=name)

def split_data(data):
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict

def check_data(vis_root, data, benchmark):
    def check_img_path(p):
        if benchmark == "OpenEQA":
            img_path = p
        else:
            img_path = os.path.join(vis_root, p)
        if not os.path.isfile(img_path):
            print(f"{img_path} not exist")
            return False
        return True

    if benchmark == "OpenEQA":
        images_path = []
        for item in data:
            path = Path(vis_root)
            sub_path = item['task_instance']['images_path'][0]
            folder = path / sub_path
            frames = sorted(folder.glob("*-rgb.png"))
            indices = np.round(np.linspace(0, len(frames) - 1, OPENEQA_NUM_FRAMES)).astype(int)
            images_path += [str(frames[i]) for i in indices]
    else:
        images_path = [p for ann in data for p in ann['task_instance']['images_path']]

    with ThreadPoolExecutor(max_workers=8) as executor:
        all_images_exist = all(executor.map(check_img_path, images_path))

    if all_images_exist:
        print('All images exist')
    else:
        exit(-1)


def update_checkboxes(selected_benchmarks):
    selected_dimensions = set()
    for benchmark in selected_benchmarks:
        selected_dimensions.update(bench_info.get(benchmark, []))
        print(f"selected dims:{selected_dimensions}")
        print(list(selected_dimensions & {"object_type", "object_property", "object_state"}),  # Object Understanding
        list(selected_dimensions & {"affordance", "world_knowledge"}),  # Knowledge
        list(selected_dimensions & {"spatial_perception", "action_perception", "temporal_perception"}),
        # Spatio-Temporal Perception
        list(selected_dimensions & {"task_related_object", "task_completion_condition", "situated_reasoning"}),
        # Reasoning
        list(selected_dimensions & {"closed-loop_planning", "open-loop_planning", "replanning"}))
    # 动态更新各个CheckboxGroup的选项
    return (
        gr.update(choices=list(selected_dimensions & {"object_type", "object_property", "object_state"})),  # Object Understanding
        gr.update(choices=list(selected_dimensions & {"affordance", "world_knowledge"})),  # Knowledge
        gr.update(choices=list(selected_dimensions & {"spatial_perception", "action_perception", "temporal_perception"})),
        # Spatio-Temporal Perception
        gr.update(choices=list(selected_dimensions & {"task_related_object", "task_completion_condition", "situated_reasoning"})),
        # Reasoning
        gr.update(choices=list(selected_dimensions & {"closed-loop_planning", "open-loop_planning", "replanning"}))  # Task Planning
    )





class EDataset(Dataset):
    def __init__(
            self, annoation, task_instructions, img_dir, benchmark, model, IMAGE_SIZE=(224,224)
    ):
        self.img_dir = img_dir
        self.annotation = annoation
        self.task_instructions = task_instructions
        self.benchmark = benchmark
        self.IMAGE_SIZE = IMAGE_SIZE
        self.model = model

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        task_instruction = self.task_instructions[ann['task_instruction_id']]
        context = task_instruction + ann['task_instance']['context']
        raw_img_list = []

        if self.model == "minigpt4":
            replace_img_token = "<ImageHere>"
        else:
            replace_img_token = ""

        has_token = False
        for i in range(len(ann['task_instance']['images_path'])):
            rmv_txt = '{image#%d}' % (i + 1)
            if rmv_txt in context:
                has_token = True
            context = context.replace(rmv_txt, replace_img_token)

        if not has_token:
            context = replace_img_token*(len(ann['task_instance']['images_path'])) + context

        if self.benchmark == "OpenEQA":
            path = Path(self.img_dir)
            sub_path = ann['task_instance']['images_path'][0]

            folder = path / sub_path
            frames = sorted(folder.glob("*-rgb.png"))
            indices = np.round(np.linspace(0, len(frames) - 1, OPENEQA_NUM_FRAMES)).astype(int)
            images_path = [str(frames[i]) for i in indices]
        else:
            images_path = ann['task_instance']['images_path']

        # first_img_size = None

        for i, p in enumerate(images_path):
            if self.benchmark == "OpenEQA":
                raw_img = Image.open(p).convert('RGB').resize(self.IMAGE_SIZE)
            else:
                img_path = os.path.join(self.img_dir, p)
                # raw_img = Image.open(img_path).convert('RGB')
                raw_img = Image.open(img_path).convert('RGB').resize(self.IMAGE_SIZE)

            # if i==0:
            #     first_img_size = raw_img.size
            # else:
            #     raw_img = raw_img.resize(first_img_size)

            raw_img_list.append(raw_img)
        return {
            "sample_id": ann['sample_id'],
            "context": context,
            "raw_img_list": raw_img_list,
            "response": str(ann['response'])
        }


class VDataset(Dataset):
    def __init__(
            self, annoation, benchmark, img_dir,
    ):
        self.img_dir = img_dir
        self.annotation = annoation
        self.benchmark = benchmark

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        ### TODO clip for EgoPlan
        if self.benchmark == "EgoPlan-Bench":
            clip = os.path.join(self.img_dir, str(ann['sample_id'])+".mp4")
        elif self.benchmark == "EgoTaskQA":
            clip = os.path.join(self.img_dir, ann['interval']+".mp4")
        else:
            clip = os.path.join(self.img_dir, ann['video_path'])

        return {
            "sample_id": ann['sample_id'],
            "question": ann['task_instance']['context'],
            "clip": clip,
            "answer": str(ann['response'])
        }


def collate_fn(batch):
    batch_data = {}
    batch_data['sample_id'] = [sample['sample_id'] for sample in batch]
    batch_data['context'] = [sample['context'] for sample in batch]
    batch_data['raw_img_list'] = [sample['raw_img_list'] for sample in batch]
    batch_data['response'] = [sample['response'] for sample in batch]

    return batch_data

# def infer_dataset_old(state, benchmark, model, benchmark_path, result_path, batch_size, progress=gr.Progress(track_tqdm=True)):
#     dataset_dir = benchmark # os.path.join(benchmark_path, benchmark)
#     img_dir = os.path.join(dataset_dir, "core/images")
#     data_path = os.path.join(dataset_dir, "core/data.json")
#     output_dir = os.path.join(result_path, model, benchmark.split("/")[-1])
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     if not os.path.exists(data_path):
#         print(f"wrong path {data_path}!!!!!!")
#         return
#
#     test_annotation = json.load(open(data_path, 'r'))
#     data_dict = split_data(test_annotation['data'])
#     check_data(img_dir, test_annotation['data'])
#     for n_img, sub_data in data_dict.items():
#         print('Checking %d-length images samples | Num:%d' % (n_img, len(sub_data)))
#
#     preds = []
#
#     try:
#         for n_img, sub_data in data_dict.items():
#             print('Proceeding %d-length images samples | Num:%d' % (n_img, len(sub_data)))
#             E = EDataset(sub_data, test_annotation['metadata']['task_instruction'], img_dir)
#             data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=int(batch_size / n_img) + 1, shuffle=False,
#                                                       num_workers=8, collate_fn=collate_fn)
#             # start_time = time.time()
#             for i, samples in enumerate(progress.tqdm(data_loader, desc='Inferring %s: Proceeding %d-length images samples | Num:%d' % (benchmark.split("/")[-1], n_img, len(sub_data)))):
#                 # pred_responses = chat.batch_answer(batch_raw_img_list=samples['raw_img_list'],
#                 #                                    batch_context=samples['context'])
#                 state['get_input'] = True
#                 image_list = []
#                 for image in samples['raw_img_list']:
#                     image_list.append(np.array(image, dtype='uint8').tolist())
#                 print(f"server_demo2 inferdataset image_list:{len(image_list)}")
#                 print(f"server_demo2 inferdataset samples['context']:{len(samples['context'])}\n{samples['context']}")
#                 state['visual'] = image_list
#                 state['data_type'] = 'image'
#                 state['text'] = samples['context']
#                 state['VLP_names'] = [model]
#
#                 model_worker_addrs = [get_model_worker_addr(model_name) for model_name in state['VLP_names']]
#                 vlp_outputs = get_model_worker_output((model_worker_addrs[0], state))
#                 state['VLP_outputs'] = vlp_outputs
#                 pred_responses = vlp_outputs
#                 print(f"vlp_outputs:{vlp_outputs}")
#
#                 for sid, gt, p in zip(samples['sample_id'], samples['response'], pred_responses):
#                     if torch.is_tensor(sid):
#                         sid = sid.item()
#                     preds.append({'sample_id': sid, 'pred_response': p, 'gt_response': gt})
#
#     finally:
#         with open(os.path.join(output_dir, 'pred.json'), 'w', encoding='utf8') as f:
#             preds.sort(key=lambda x: x["sample_id"])
#             json.dump(preds, f, indent=4, ensure_ascii=False)


def infer_dataset(state, benchmark, dim, model, benchmark_root, result_path, batch_size=1, progress=gr.Progress(track_tqdm=True)):
    dataset_dir = os.path.join(benchmark_root, benchmark, dim) # os.path.join(benchmark_path, benchmark)
    if benchmark == "OpenEQA":
        img_dir = os.path.join(benchmark_root, benchmark, "images")
    else:
        img_dir = os.path.join(dataset_dir, "core/images")
    data_path = os.path.join(dataset_dir, "core/data.json")
    output_dir = os.path.join(result_path, model, benchmark, dim)

    if not os.path.exists(dataset_dir):
        print(f"\n*****************\n Error: no such data: {dataset_dir}!!!!!!!!!!! \n*****************")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(data_path):
        print(f"wrong path {data_path}!!!!!!")
        return

    test_annotation = json.load(open(data_path, 'r'))
    data_dict = split_data(test_annotation['data'])
    check_data(img_dir, test_annotation['data'], benchmark)
    for n_img, sub_data in data_dict.items():
        print('Checking %d-length images samples | Num:%d' % (n_img, len(sub_data)))

    preds = []

    try:
        for n_img, sub_data in data_dict.items():
            print('Proceeding %d-length images samples | Num:%d' % (n_img, len(sub_data)))
            E = EDataset(sub_data, test_annotation['metadata']['task_instruction'], img_dir, benchmark, model)
            data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=int(batch_size / n_img) + 1, shuffle=False,
                                                      num_workers=8, collate_fn=collate_fn)
            # start_time = time.time()
            for i, samples in enumerate(progress.tqdm(data_loader, desc='Inferring %s-%s: Proceeding %d-length images samples | Num:%d' % (benchmark, dim, n_img, len(sub_data)))):
                # pred_responses = chat.batch_answer(batch_raw_img_list=samples['raw_img_list'],
                #                                    batch_context=samples['context'])
                state['get_input'] = True
                image_list_list = []
                for image in samples['raw_img_list']:
                    image_list_list.append(np.array(image, dtype='uint8').tolist())
                print(f"server_demo2 inferdataset sample_id:{samples['sample_id']}")
                print(f"server_demo2 inferdataset image_list_list:{len(image_list_list)}")
                print(f"server_demo2 inferdataset samples['context']:{len(samples['context'])}\n{samples['context']}")
                state['visual'] = image_list_list
                state['data_type'] = 'image'
                state['text'] = samples['context']
                state['VLP_names'] = [model]

                model_worker_addrs = [get_model_worker_addr(model_name) for model_name in state['VLP_names']]
                vlp_outputs = get_model_worker_output((model_worker_addrs[0], state))
                state['VLP_outputs'] = vlp_outputs
                if isinstance(vlp_outputs, list):
                    pred_responses = vlp_outputs
                else:
                    pred_responses = [vlp_outputs]
                print(f"vlp_outputs:{vlp_outputs}")

                for sid, gt, p in zip(samples['sample_id'], samples['response'], pred_responses):
                    if torch.is_tensor(sid):
                        sid = sid.item()
                    preds.append({'sample_id': sid, 'pred_response': p, 'gt_response': gt})

    finally:
        with open(os.path.join(output_dir, 'pred.json'), 'w', encoding='utf8') as f:
            print(json.dumps(preds))
            preds.sort(key=lambda x: x["sample_id"])
            json.dump(preds, f, indent=4, ensure_ascii=False)


def infer_dataset_video(state, benchmark, dim, model, benchmark_root, result_path, batch_size=1, progress=gr.Progress(track_tqdm=True)):
    dataset_dir = os.path.join(benchmark_root, benchmark, dim) # os.path.join(benchmark_path, benchmark)
    img_dir = os.path.join(benchmark_root, benchmark, "videos")
    data_path = os.path.join(dataset_dir, "core/data.json")
    output_dir = os.path.join(result_path, model, benchmark, dim)

    if not os.path.exists(dataset_dir):
        print(f"\n*****************\n Error: no such data: {dataset_dir}!!!!!!!!!!! \n*****************")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(data_path):
        print(f"wrong path {data_path}!!!!!!")
        return

    # test_annotation = json.load(open(data_path, 'r'))
    # data_dict = split_data(test_annotation['data'])
    test_annotation = json.load(open(data_path, 'r'))
    # check_data(img_dir, test_annotation['data'])

    preds = []

    try:
        E = VDataset(test_annotation["data"], benchmark, img_dir)
        data_loader = torch.utils.data.DataLoader(dataset=E, batch_size=int(batch_size), shuffle=False, num_workers=8)
        # start_time = time.time()
        for i, samples in enumerate(progress.tqdm(data_loader, desc='Inferring %s-%s: Proceeding samples | Num:%d' % (benchmark, dim, len(test_annotation["data"])))):
            # pred_responses = chat.batch_answer(batch_raw_img_list=samples['raw_img_list'],
            #                                    batch_context=samples['context'])
            state['get_input'] = True

            print(f"server_demo2 inferdataset sample_id:{samples['sample_id']}")
            # print(f"server_demo2 inferdataset image_list:{len(image_list)}")
            print(f"server_demo2 inferdataset samples['question']:{len(samples['question'])}\n{samples['question']}")
            state['visual'] = samples['clip']
            print(samples['clip'])
            state['data_type'] = 'video'
            state['text'] = samples['question']
            state['VLP_names'] = [model]
            # print(f"model___name:{model}")

            model_worker_addrs = [get_model_worker_addr(model_name) for model_name in state['VLP_names']]
            vlp_outputs = get_model_worker_output((model_worker_addrs[0], state))
            state['VLP_outputs'] = vlp_outputs
            if isinstance(vlp_outputs, list):
                pred_responses = vlp_outputs
            else:
                pred_responses = [vlp_outputs]
            print(f"vlp_outputs:{vlp_outputs}")

            for sid, gt, p in zip(samples['sample_id'], samples['answer'], pred_responses):
                if torch.is_tensor(sid):
                    sid = sid.item()
                preds.append({'sample_id': sid, 'pred_response': p, 'gt_response': gt})

    finally:
        with open(os.path.join(output_dir, 'pred.json'), 'w', encoding='utf8') as f:
            print(json.dumps(preds))
            preds.sort(key=lambda x: x["sample_id"])
            json.dump(preds, f, indent=4, ensure_ascii=False)



def evaluate(state, benchmarks, obj_understanding, knowledge, spatio_temporal, reasoning, task_planning, model, benchmark_path, result_path, batch_size, progress=gr.Progress(track_tqdm=True)):
    dims = obj_understanding + knowledge + spatio_temporal + reasoning + task_planning

    for i, dim in enumerate(dims):
        dims[i] = dim.lower().replace(" ", "_")

    if state["data_type"] == "image":
        for bench in benchmarks:
            print(f"On {bench} Benchmark:")
            for dim in dims:
                if dim in bench_info[bench]:
                    print(f"on {dim} dim...")
                    infer_dataset(state, bench, dim, model, benchmark_path, result_path, batch_size)

    # TODO: support video dataset !!!!!!!!!!!!!!
    elif state["data_type"] == "video":
        for bench in benchmarks:
            print(f"On {bench} Benchmark:")
            for dim in dims:
                if dim in bench_info[bench]:
                    print(f"on {dim} dim...")
                    infer_dataset_video(state, bench, dim, model, benchmark_path, result_path, batch_size)

    return "Inference Finished"


def update_benchmark_choices(media_type):
    if media_type == "image":
        return bench_info["img_bench_list"]
    elif media_type == "video":
        return bench_info["video_bench_list"]
    return []


def toggle_layout(state, data_type, video_model_cnt):
    if video_model_cnt>0:
        if data_type=="image":
            state['data_type'] = "image"
            return state, gr.update(visible=True), gr.update(visible=False), gr.update(choices=bench_info["img_bench_list"], value=bench_info["img_bench_list"])
        else:
            state['data_type'] = "video"
            return state, gr.update(visible=False), gr.update(visible=True), gr.update(choices=bench_info["video_bench_list"], value=bench_info["video_bench_list"])
    else:
        if data_type=="image":
            state['data_type'] = "image"
            return state, gr.update(visible=True), gr.update(choices=bench_info["img_bench_list"], value=bench_info["img_bench_list"])
        else:
            state['data_type'] = "video"
            return state, gr.update(visible=False), gr.update(choices=bench_info["video_bench_list"], value=bench_info["video_bench_list"])

def get_video_model(model_list):
    video_model_list = []
    for video_model in video_models:
        if video_model in model_list:
            video_model_list.append(video_model)

    return video_model_list


def select_all():
    return (["object_type", "object_property", "object_state"],
            ["affordance", "world_knowledge"],
            ["spatial_perception", "action_perception", "temporal_perception"],
            ["task_related_object", "task_completion_condition", "situated_reasoning"],
            ["closed-loop_planning", "open-loop_planning", "replanning"])

def deselect_all():
    return [], [], [], [], []


def find_model_data(rank_data, model_name):
    # print(type(rank_data), rank_data)
    for i, data in rank_data.iterrows():
        # print(type(data), data)
        if data["Model"] == model_name:
            return i

    return 100000

def plot_fig(table_id, rank_data1, rank_data2, dims, models, fig_type):
    color_list = ['green', 'cornflowerblue', 'salmon', 'black', 'goldenrod', 'mediumpurple', 'sienna', 'cyan', 'plum']
    results = []

    if table_id == "Tab1":
        rank_data = rank_data1
    elif table_id == "Tab2":
        rank_data = rank_data2

    for i, model_name in enumerate(models):
        results.append({})
        for j, dim in enumerate(dims):
            results[i][dim] = rank_data.iloc[find_model_data(rank_data, model_name)][dim]
    if fig_type == "radar":
        data_length = len(results[0])
        angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)
        labels = [key for key in results[0].keys()]
        score = [[v for v in result.values()] for result in results]
        score_list = []
        for i in range(len(score)):
            score_list.append(np.concatenate((score[i], [score[i][0]])))

        angles = np.concatenate((angles, [angles[0]]))
        labels = np.concatenate((labels, [labels[0]]))
        # fig = plt.figure(figsize=(40, 30), dpi=200)
        fig = plt.figure(figsize=(8,6), dpi=200)
        ax = plt.subplot(111, polar=True)
        for i in range(len(score_list)):
            ax.plot(angles, score_list[i], color=color_list[i], alpha=0.7, lw=2, label=models[i])
            if color_list[i] == 'black':
                ax.fill(angles, score_list[i], color=color_list[i], alpha=0.1)
            else:
                ax.fill(angles, score_list[i], color=color_list[i], alpha=0.2)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.set_theta_zero_location('N')
        ax.set_rlim(0, 1)
        ax.set_rlabel_position(0)
        ax.spines['polar'].set_color('gray')
        ax.spines['polar'].set_alpha(0.4)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=6, columnspacing=0.25, handlelength=1.5)
        ax.set_title('Radar Chart')
        plt.savefig("./tem_fig/output_fig.png")
        return Image.open("./tem_fig/output_fig.png")

    elif fig_type == "bar":
        bar_width = 0.08
        index = np.arange(len(dims))

        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        for i, model_name in enumerate(models):
            scores = [results[i][dim] for dim in dims]
            ax.bar(index + i * bar_width, scores, bar_width, label=model_name, color=color_list[i % len(color_list)])

        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Scores')
        ax.set_title('Bar Chart')
        ax.set_xticks(index + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(dims)
        ax.legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig("./tem_fig/output_fig.png")
        return Image.open("./tem_fig/output_fig.png")


def calculate_metric(benchmarks, model, metric_type, benchmark_path, result_path, obj_understanding, knowledge, spatio_temporal, reasoning, task_planning):
    ### TODO: handle abs path
    result_root = result_path
    dataset_root = benchmark_path
    E = Eval()
    dims = obj_understanding + knowledge + spatio_temporal + reasoning + task_planning
    if metric_type == "model-based":
        print("not supported metric type!")
        exit(-1)
    elif metric_type == "rule-based":
        for benchmark in benchmarks:
            result_path = os.path.join(result_root, model, benchmark)
            dataset_path = os.path.join(dataset_root, benchmark)

            for dim in bench_info[benchmark]:
                if dim in dims:
                    try:
                        print(model, end=':  ')
                        print(dim, end=':  ')
                        if not os.path.exists(os.path.join(result_path, dim, 'pred.json')):
                            print('%s--%s--%s  No prediction file found' % (model, benchmark, dim))
                            print(os.path.join(result_path, dim, 'pred.json'))
                            return '%s--%s--%s  No prediction file found' % (model, benchmark, dim)
                        core_annotation = json.load(open(os.path.join(dataset_path, dim, 'core', 'data.json'), 'r', encoding='utf-8'))
                        question_type = core_annotation['metadata']['question_type']
                        preds = json.load(open(os.path.join(result_path, dim, 'pred.json'), 'r', encoding='utf-8'))

                        if question_type == 'open-ended':
                            eval_result, eval_list = E.evaluate_rouge(preds)
                        elif question_type == 'multi-choice':
                            eval_result, eval_list = E.evaluate_multichoice(preds, core_annotation)
                        elif question_type == 'multi-answer':
                            eval_result, eval_list = E.evaluate_multi_answer(preds, core_annotation)
                        else:
                            print('Dataset not supported')
                            exit(-1)

                        with open(os.path.join(result_path, dim, 'eval.json'), 'w') as f:
                            json.dump(eval_result, f)

                        with open(os.path.join(result_path, dim, 'eval_score.json'), 'w') as f:
                            json.dump(eval_list, f, indent=4)
                    except Exception as e:
                        print(f"Error occured when calculating metric of {benchmark}-{dim}:\n {e}\nPlease check the result files!")
    return f"Calculate metric finished"


def build_demo(show_arena, benchmark_path, result_path):

    with gr.Blocks(theme=gr.themes.Soft(), title='EmbodiedEval') as demo:

        state = gr.State({'data_type':'image'})
        video_model_list = get_video_model(model_list)
        with gr.Tab("Direct Chat"):
            # with gr.Row():
            #     gr.HTML(open("CVLAB/header.html", "r").read())
            gr.Markdown("""# 🤖 Direct Chat with Models
                           ## 👇 Choose a model to chat""")

            # video_model_list = get_video_model(model_list)
            video_model_cnt = gr.Number(len(video_model_list), visible=False)
            if video_model_list:
                data_type = gr.Radio(["image", "video"], value="image", label="Media type")
                radio = gr.Radio(model_list, value=model_list[0], label="Model", info="current available model(s):")
                radio_video = gr.Radio(video_model_list, value=video_model_list[0], label="Video Model", info="current available video model(s):", visible=False)

            else:
                data_type = gr.Radio(["image"], value="image", label="Media type")
                radio = gr.Radio(model_list, value=model_list[0], label="Model", info="current available model(s):")

            with gr.Group():
                with gr.Row():
                    # with gr.Column(scale=1):
                    #     videobox = gr.Video(visible=False)
                    with gr.Column():
                        model_name_A = gr.Markdown("")
                        chatbot_A = gr.Chatbot(label=model_list[0])
                with gr.Group():
                    with gr.Row():
                        good_btn = gr.Button(value="👍  Good", interactive=False, visible=False)
                        bad_btn = gr.Button(value="👎  Bad", interactive=False, visible=False)

            with gr.Row():
                with gr.Column(scale=20):
                    textbox = gr.MultimodalTextbox(
                        show_label=False,
                        placeholder="Enter text and image/video here, and press ENTER to send",
                        container=False
                    )
                # with gr.Column():  # with gr.Column(scale=1, min_width=50):
                send_btn = gr.Button(value="Send", visible=False)

            with gr.Row():
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
                share_btn = gr.Button(value="📷  Share")

            gr.Examples(examples=[
                [{'text': 'What am I doing?', 'files': [f'benchmarks/EgoThink/action_perception/core/images/0.jpg']}],
                [{'text': 'Can this object in my hand be used to hold water?', 'files': [f'benchmarks/EgoThink/affordance/core/images/0.jpg']}],
                [{'text': 'What is the object enclosed by a red box?', 'files': [f'benchmarks/MFE-ETP/object_type/core/images/bottling_fruit_0.jpg']}],
                [{'text': 'Arrange these three photos in order from farthest to nearest based on the distance between the camera and the table, like "image1(farthest), image2(secondary), image3(nearest)"',
                  'files': [f'benchmarks/MFE-ETP/spatial_perception/core/images/collect_misplaced_items_0.jpg',
                            f'benchmarks/MFE-ETP/spatial_perception/core/images/collect_misplaced_items_1.jpg',
                            f'benchmarks/MFE-ETP/spatial_perception/core/images/collect_misplaced_items_2.jpg',]}]
            ], inputs=textbox)

            # example_video = gr.Examples(examples=[
            #     [f"examples/cat.mp4", "What is in this video?"]
            # ], inputs=[textbox],vi)

            gr.Markdown(notice_markdown)
            gr.Markdown(license_markdown)

            chat_list = [chatbot_A]
            model_name_list = [model_name_A]
            state_list = [state] + chat_list
            vote_list = [good_btn, bad_btn]
            btn_list = vote_list + [regenerate_btn, clear_btn, send_btn]

            if video_model_list:
                data_type.change(toggle_layout, [state, data_type, video_model_cnt], [state]+[radio, radio_video])
                radio_video.change(change_name, radio_video, chatbot_A)
            else:
                data_type.change(toggle_layout, [state, data_type, video_model_cnt], [state, radio])
            radio.change(change_name, radio, chatbot_A)

            good_btn.click(vote_model_good_single, [state] + model_name_list, model_name_list + vote_list)
            bad_btn.click(vote_model_bad_single, [state] + model_name_list, model_name_list + vote_list)
            clear_btn.click(clear_chat_single, [state], state_list + model_name_list + [textbox] + btn_list)
            share_btn.click(share_click, [state], [state])
            regenerate_btn.click(run_VLP_models_single, state_list, state_list + btn_list)
            if video_model_list:
                textbox.submit(user_ask_single, state_list + [textbox, radio, radio_video], state_list + model_name_list + [textbox] + btn_list).then(run_VLP_models_single, state_list, state_list + btn_list)
                send_btn.click(user_ask_single, state_list + [textbox, radio, radio_video], state_list + model_name_list + [textbox] + btn_list).then(run_VLP_models_single, state_list, state_list + btn_list)
            else:
                textbox.submit(user_ask_single, state_list + [textbox, radio], state_list + model_name_list + [textbox] + btn_list).then(run_VLP_models_single, state_list, state_list + btn_list)
                send_btn.click(user_ask_single, state_list + [textbox, radio], state_list + model_name_list + [textbox] + btn_list).then(run_VLP_models_single, state_list, state_list + btn_list)


        if show_arena:
            with gr.Tab("Arena"):
                # with gr.Row():
                #     gr.HTML(open("CVLAB/header.html", "r").read())
                gr.Markdown("""# ⚔️ EmbodiedEval Arena""")
                gr.Markdown(rules_markdown)
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1):
                            imagebox = gr.Image(type="pil")
                        with gr.Column():
                            model_name_A = gr.Markdown("")
                            chatbot_A = gr.Chatbot(label='Model A', height=550)
                        with gr.Column():
                            model_name_B = gr.Markdown("")
                            chatbot_B = gr.Chatbot(label='Model B', height=550)
                    with gr.Group():
                        with gr.Row():
                            leftvote_btn = gr.Button(value="👈  A is better", interactive=False)
                            rightvote_btn = gr.Button(value="👉  B is better", interactive=False)
                            tie_btn = gr.Button(value="🤝  Tie", interactive=False)
                            bothbad_btn = gr.Button(value="👎  Both are bad", interactive=False)

                with gr.Row():
                    with gr.Column(scale=20):
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            container=False
                        )
                    with gr.Column(scale=1, min_width=50):
                        send_btn = gr.Button(value="Send")

                with gr.Row():
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
                    share_btn = gr.Button(value="📷  Share")

                gr.Examples(examples=[
                    [f"examples/merlion.png", "Which city is this?"],
                    [f"examples/kun_basketball.jpg", "Is the man good at playing basketball?"],
                    [f"examples/tiananmen.jpg", "Which country this image describe?"]
                ], inputs=[imagebox, textbox])
                gr.Markdown(notice_markdown)
                gr.Markdown(license_markdown)
                chat_list = [chatbot_A, chatbot_B]
                model_name_list = [model_name_A, model_name_B]
                state_list = [state] + chat_list
                vote_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn]
                btn_list = vote_list + [regenerate_btn, clear_btn, send_btn]
                leftvote_btn.click(vote_left_model, [state] + model_name_list, model_name_list + vote_list)
                rightvote_btn.click(vote_right_model, [state] + model_name_list, model_name_list + vote_list)
                tie_btn.click(vote_model_tie, [state] + model_name_list, model_name_list + vote_list)
                bothbad_btn.click(vote_model_bad, [state] + model_name_list, model_name_list + vote_list)
                clear_btn.click(clear_chat, [state], state_list + model_name_list + [textbox, imagebox] + btn_list)
                share_btn.click(share_click, [state], [state])
                regenerate_btn.click(run_VLP_models, state_list, state_list + btn_list)
                textbox.submit(user_ask, state_list + [textbox, imagebox], state_list + model_name_list + [textbox, imagebox] + btn_list).then(run_VLP_models, state_list, state_list + btn_list)
                send_btn.click(user_ask, state_list + [textbox, imagebox], state_list + model_name_list + [textbox, imagebox] + btn_list).then(run_VLP_models, state_list, state_list + btn_list)


        with gr.Tab("Evaluate"):
            gr.Markdown("""# 💡 Evaluate Models on Benchmarks""")

            with gr.Row():
                # radio = gr.Radio(model_list, value=model_list[0], label="Model", info="current available model(s):")
                video_model_cnt = gr.Number(len(video_model_list), visible=False)
                if video_model_list:
                    data_type = gr.Radio(["image", "video"], value="image", label="Media type")
                    radio = gr.Radio(model_list, value=model_list[0], label="Model", info="current available model(s):")
                    radio_video = gr.Radio(video_model_list, value=video_model_list[0], label="Video Model",
                                           info="current available video model(s):", visible=False)

                else:
                    data_type = gr.Radio(["image"], value="image", label="Media type")
                    radio = gr.Radio(model_list, value=model_list[0], label="Model", info="current available model(s):")
                batch_size = gr.Slider(1, 16, value=1, step=1, label="Batch Size", visible=False)
            benchmark = gr.Textbox(value=benchmark_path, visible=False)
            result = gr.Textbox(value=result_path, visible=False)
            with gr.Group():

                with gr.Row():
                    select_all_button = gr.Button("Select All")
                    deselect_all_button = gr.Button("Deselect All")

                with gr.Row():
                    with gr.Column():
                        benchmarks_list = bench_info["img_bench_list"]

                        benchmarks = gr.CheckboxGroup(
                            choices=benchmarks_list,
                            value=benchmarks_list,
                            label="Select Benchmark"
                        )

                        with gr.Accordion("Object Understanding", open=True):
                            obj_understanding = gr.CheckboxGroup(
                                choices=["object_type", "object_property", "object_state"],
                                label="Select sub-dimensions"
                            )

                        with gr.Accordion("Knowledge", open=True):
                            knowledge = gr.CheckboxGroup(
                                choices=["affordance", "world_knowledge"],
                                label="Select sub-dimensions"
                            )

                    with gr.Column():
                        with gr.Accordion("Spatio-Temporal Perception", open=True):
                            spatio_temporal = gr.CheckboxGroup(
                                choices=["spatial_perception", "action_perception", "temporal_perception"],
                                label="Select sub-dimensions"
                            )

                        with gr.Accordion("Reasoning", open=True):
                            reasoning = gr.CheckboxGroup(
                                choices=["task_related_object", "task_completion_condition", "situated_reasoning"],
                                label="Select sub-dimensions"
                            )

                        with gr.Accordion("Task Planning", open=True):
                            task_planning = gr.CheckboxGroup(
                                choices=["closed-loop_planning", "open-loop_planning", "replanning"],
                                label="Select sub-dimensions"
                            )

                submit_btn = gr.Button("Start Inference", scale=1)

            if video_model_list:
                data_type.change(toggle_layout, [state, data_type, video_model_cnt], [state]+[radio, radio_video, benchmarks])
                radio_video.change(change_name, radio_video, chatbot_A)
            else:
                data_type.change(toggle_layout, [state, data_type, video_model_cnt], [state, radio]+[benchmarks])
            progress_box = gr.Textbox(label="Inference Status", value="Press 'Start Inference' to start the inference with chosen model and data")

            with gr.Row():
                with gr.Column():
                    metric_btn = gr.Button("Calculate metric")
                    metric_type = gr.Radio(["model-based", "rule-based"], value="rule-based", label="choose a metric:")


                with gr.Column():
                    metric_box = gr.Textbox(label="Metric Status", value="Press 'Calculate metric' to calculate metric score on chosen data")

            metric_btn.click(fn=calculate_metric,
                             inputs=[benchmarks, radio, metric_type, benchmark, result, obj_understanding, knowledge, spatio_temporal, reasoning, task_planning],
                             outputs=metric_box)
            benchmarks.change(
                update_checkboxes,
                inputs=[benchmarks],
                outputs=[obj_understanding, knowledge, spatio_temporal, reasoning, task_planning]
            )
            submit_btn.click(fn=evaluate, inputs=[state, benchmarks, obj_understanding, knowledge, spatio_temporal, reasoning, task_planning, radio, benchmark, result, batch_size], outputs=progress_box)
            select_all_button.click(
                select_all,
                outputs=[obj_understanding, knowledge, spatio_temporal, reasoning, task_planning]
            )

            deselect_all_button.click(
                deselect_all,
                outputs=[obj_understanding, knowledge, spatio_temporal, reasoning, task_planning]
            )
            gr.Markdown(notice_markdown)
            gr.Markdown(license_markdown)


        with gr.Tab("Leaderboard"):
            gr.Markdown("""# 🏅 EmbodiedEval Leaderboard""")
            gr.Markdown("""Leaderboard description""")
            with gr.Row():
                with gr.Tab("Tab1"):
                    rank_data1 = gr.Dataframe(
                        label="Leaderboard1",
                        headers=["Rank", "Model", "Total score", "Object Understanding", "Knowledge",
                                 "Spatio-Temporal Perception", "Reasoning", "Task Planning"],
                        datatype=["number", "str", "number", "number", "number", "number", "number", "number"],
                        value=[[1, "GPT-4V", 0.67, 0.4, 0.3, 0.5, 0.6, 0.2],
                               [2, "LLaMA-VID", 0.61, 0.4, 0.3, 0.4, 0.5, 0.1],
                               [3, "MiniCPM", 0.56, 0.3, 0.3, 0.4, 0.5, 0.1],
                               [4, "Video-ChatGPT", 0.55, 0.25, 0.3, 0.4, 0.48, 0.1],
                               [5, "Video-LLaVA", 0.49, 0.25, 0.3, 0.35, 0.45, 0.1],
                               [6, "LLaVA", 0.45, 0.2, 0.3, 0.35, 0.45, 0.1],
                               [7, "InstructBLIP", 0.4, 0.2, 0.3, 0.3, 0.45, 0.1],
                               [8, "MiniGPT-4", 0.39, 0.18, 0.3, 0.35, 0.4, 0.1],
                               [9, "BLIP-2", 0.38, 0.15, 0.3, 0.30, 0.4, 0.1],
                               ],
                        # row_count=5,
                        col_count=(8, "fixed"),
                        interactive=False
                    )
                with gr.Tab("Tab2"):
                    rank_data2 = gr.Dataframe(
                        label="Leaderboard2",
                        headers=["Rank", "Model", "Total score", "object type", "object property", "object state"
                                 "affordance", "world knowledge", "spatial_perception", "action_perception", "temporal_perception", "task_related_object", "task_completion_condition", "situated_reasoning", "closed-loop_planning", "open-loop_planning", "replanning"],
                        datatype=["number", "str", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number"],
                        value=[[1, "GPT-4V", 0.75, 0.23, 0.81, 0.67, 0.12, 0.48, 0.93, 0.54, 0.39, 0.16, 0.67, 0.82, 0.15, 0.99],
                                [2, "LLaMA-VID", 0.13, 0.91, 0.68, 0.42, 0.37, 0.79, 0.56, 0.47, 0.29, 0.85, 0.31, 0.55, 0.92, 0.92],
                                [3, "MiniCPM", 0.33, 0.67, 0.14, 0.88, 0.75, 0.20, 0.92, 0.44, 0.58, 0.91, 0.24, 0.60, 0.18, 0.92],
                                [4, "Video-ChatGPT", 0.58, 0.12, 0.95, 0.71, 0.36, 0.61, 0.47, 0.30, 0.19, 0.54, 0.71, 0.90, 0.74, 0.92],
                                [5, "Video-LLaVA", 0.49, 0.71, 0.35, 0.52, 0.17, 0.68, 0.34, 0.48, 0.22, 0.83, 0.62, 0.94, 0.59, 0.92],
                                [6, "LLaVA", 0.91, 0.30, 0.15, 0.65, 0.39, 0.87, 0.21, 0.53, 0.78, 0.27, 0.62, 0.40, 0.98, 0.92],
                                [7, "InstructBLIP", 0.24, 0.88, 0.70, 0.19, 0.61, 0.50, 0.86, 0.43, 0.53, 0.84, 0.33, 0.20, 0.99, 0.92],
                                [8, "MiniGPT-4", 0.77, 0.42, 0.69, 0.83, 0.19, 0.51, 0.34, 0.64, 0.59, 0.72, 0.16, 0.93, 0.48, 0.92],
                                [9, "BLIP-2", 0.60, 0.77, 0.12, 0.28, 0.41, 0.63, 0.47, 0.50, 0.87, 0.35, 0.79, 0.91, 0.65, 0.92]
                               ],
                        # row_count=5,
                        col_count=(16, "fixed"),
                        interactive=False
                    )


            gr.Markdown("""# 📊 Analyze the Results""")
            # gr.Markdown("""xxxxxxxxxxxxx""")
            # with gr.Group():

                # gr.Interface(
                #     plot_fig,
                #     [
                #         rank_data,
                #         gr.CheckboxGroup(["Total score", "Object Understanding", "Knowledge",
                #              "Spatio-Temporal Perception", "Reasoning", "Task Planning"], label="Score type:"),
                #         gr.CheckboxGroup(["LLaVA", "BLIP-2", "MiniGPT-4", "InstructBLIP", "MiniCPM", "GPT-4V", "Video-LLaVA", "Video-ChatGPT", "LLaMA-VID"], label="Model Selection"),
                #         # gr.Slider(1, 100, label="Noise Level"),
                #         # gr.Checkbox(label="Show Legend"),
                #         gr.Dropdown(["bar", "radar"], value="radar", label="Style"),
                #     ],
                #     gr.Plot(label="forecast"), #  format="png",
                #     allow_flagging=False
                # )
            with gr.Row():
                with gr.Column():
                    tab_selecter = gr.Radio(["Tab1", "Tab2"], value="Tab1", label="data table")
                    score_type = gr.CheckboxGroup(["Total score", "Object Understanding", "Knowledge",
                                      "Spatio-Temporal Perception", "Reasoning", "Task Planning"],
                                                  value=["Object Understanding", "Knowledge",
                                      "Spatio-Temporal Perception", "Reasoning", "Task Planning"],
                                     label="Score type:")

                    models = gr.CheckboxGroup(
                        ["LLaVA", "BLIP-2", "MiniGPT-4", "InstructBLIP", "MiniCPM", "GPT-4V", "Video-LLaVA",
                         "Video-ChatGPT", "LLaMA-VID"], label="Model Selection")

                    fig_type = gr.Dropdown(["bar", "radar"], value="radar", label="Style")

                    # gr.Slider(1, 100, label="Noise Level"),
                    # gr.Checkbox(label="Show Legend"),

                with gr.Column():
                    output_fig = gr.Image()

            score_type.change(plot_fig, [tab_selecter, rank_data1, rank_data2, score_type, models, fig_type], output_fig)
            models.change(plot_fig, [tab_selecter, rank_data1, rank_data2, score_type, models, fig_type], output_fig)
            fig_type.change(plot_fig, [tab_selecter, rank_data1, rank_data2, score_type, models, fig_type], output_fig)

            gr.Markdown(notice_markdown)
            gr.Markdown(license_markdown)



    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--controller-url", type=str, default="http://localhost:12001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--benchmark_path", type=str, default="./benchmarks")
    parser.add_argument("--result_path", type=str, default="./results")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller_url = args.controller_url
    model_list = get_model_list(controller_url)
    model_list = ["llava", "blip2", "minigpt4", "instructblip", "minicpm", "videollava", "video-chatgpt", "llama-vid", "gpt-4v"]
    print(f"Available model: {', '.join(model_list)}")
    assert len(model_list) >= 1, "Available model number should not smaller than 1."
    show_arena = False
    if len(model_list) >= 2:
        print("building demo with arena")
        show_arena = True
    else:
        print("building demo without arena")


    # TODO: arena for future use
    show_arena=False

    demo = build_demo(show_arena, args.benchmark_path, args.result_path)
    demo.queue(
        default_concurrency_limit=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )