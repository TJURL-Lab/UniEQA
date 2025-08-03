import os
import re
import threading

from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import requests
import json
import sys
import time
import traceback

import gradio as gr


def read_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
    return config


cfg = read_config("./peng_utils/openai_cfg.json")["apis"][1]

API_KEY = cfg["api_key"] 

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["http_proxy"] = cfg["http_proxy"]
os.environ["https_proxy"] = cfg["https_proxy"]

### TODO #open source: remove my Proxy
Proxy = cfg["proxy"]    # "apix.ai-gaochao.cn"    # default: Proxy = "api.openai.com" # openai.arnotho.com

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


    def get_gpt_scores(self, question, pred_ans, gt_ans, question_type, acc, eval_list, sample_id, max_tokens=300,
                       model="gpt-4o-mini"):    #
        print(f"#############################\nsample_id: {sample_id}")
        if pred_ans == '':
            s = 0
            a = "pred_ans is none"
        else:
            context_dict = {
                'open-ended': "Now, you will be presented with a correct response, and a student's answer to an open-ended question. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)', which is your final judgement. Output in JSON format. For instance: '{\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0)\"}'. The correct response and student's answer is provided below.",
                'multi-choice': "Now, you will be presented with the correct answer and a student's response to a multiple-choice question, where multiple options are provided but only one is correct. Your task is to first match the student's answer to the most relevant option and then evaluate its correctness based on the following rules: If the the most relevant option corresponds to the correct answer, assign a score of '1'. If the the most relevant option does not match the correct answer, assign a score of '0'. Begin your evaluation with an 'Assessment:' paragraph, where you explain your reasoning. Conclude with 'Final Score: 1 (or 0)' as your final judgment. Output in JSON format, for example: '{\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0)\"}'. The correct answer and the student's response are provided below.",
                'multi-answer': "Now, you will be given the correct answer(s) and a student's response to a multiple-selection question, where multiple options are provided, and one or more may be correct. Your task is to first match the student's answer to the most relevant option(s) and then evaluate its correctness based on the following rules: If the student's answer includes all correct options and no incorrect ones, assign a score of '1'. If the student's answer contains any incorrect options, assign a score of '0'. If the student's answer includes some but not all of the correct options and no incorrect ones, assign a score of '0.5'. Begin your evaluation with an 'Assessment:' paragraph, where you explain your reasoning. Conclude with 'Final Score: 1, 0.5, or 0' as your final judgment. Output in JSON format, for example: '{\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0)\"}'. The correct answer(s) and the student's response are provided below.",
                'default': "Now, you will be presented with a correct response, and a student's answer to some question. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)', which is your final judgement. Output in JSON format. For instance: {\"Assessment\": \"xxxxx\", \"Final Score\": \"1(or 0)\"}. The correct response and student's answer is provided below."
            }

            if question_type in context_dict.keys():
                context = context_dict[question_type]
                print(f"using {question_type} specific instruction")
            else:
                context = context_dict['default']
                print(f"using DEFAULT instruction")
            qa_info = f"\nThe Student’s Answer is:\"{pred_ans}\"\nThe Correct Answer is:\"{gt_ans}\""
            context = context + qa_info

            print(f"get_gpt_scores context: {context}\n")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }

            payload = {
                "model": str(model),
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You, as an AI system, have been equipped with a vast amount of knowledge and an understanding of logical reasoning. Your task is to use these capabilities to assess academic responses."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": context
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }
            ### TODO #open source: change 'apix.ai-gaochao.cn' to 'api.openai.com'
            for i in range(200):
                try:
                    if i >= 1:
                        time.sleep(0.5)
                        print(f"Retrying times: {i}")
                    response = requests.post("https://" + Proxy + "/v1/chat/completions", headers=headers,
                                             json=payload, timeout=500)

                    output = eval(response.json()['choices'][0]['message']['content'])
                    assessment = output["Assessment"]
                    score = float(output["Final Score"])

                    if response.status_code == requests.codes.ok:
                        break
                except Exception as e:
                    print(e)  # 输出：division by zero
                    print(sys.exc_info())
                    print('\n', '>>>' * 20)
                    print(traceback.print_exc())
                    print('\n', '>>>' * 20)
                    print(traceback.format_exc())
                    print(response.json())
            s = score
            a = assessment

        acc['f'].append(s)
        eval_list.append({'id': str(sample_id), 'score': str(round(s, 3)), 'assessment': a})
        print(eval_list[-1])
        return score

    def evaluate_gpt(self, preds, question_type, datas, benchmark, dim, progress=gr.Progress(track_tqdm=True)):
        acc = {'f': [], }
        eval_list = []
        NUM_PROCESS = 10
        try:
            for i in progress.tqdm(range(0, len(preds), NUM_PROCESS), desc='Evaluating %s-%s: Proceeding %d samples:' % (benchmark, dim, len(preds))):
                # mp:
                Threads = []
                for j in range(NUM_PROCESS):
                    # print(i+j)
                    if i + j >= len(preds):
                        break
                    sample_id = preds[i + j]['sample_id']
                    gt_ans = self.process(preds[i + j]["gt_response"])
                    pred_ans = self.process(preds[i + j]["pred_response"])
                    question = datas[i + j]["task_instance"]["context"]
                    assert gt_ans != ''
                    t = threading.Thread(target=self.get_gpt_scores,
                                         args=((question, pred_ans, gt_ans, question_type, acc, eval_list, sample_id)))
                    t.setDaemon = True
                    Threads.append(t)
                for t in Threads:
                    t.start()
                for t in Threads:
                    t.join()

        finally:
            results = {'gpt-4o-mini': np.mean(acc['f'])}
            eval_list.sort(key=lambda x: int(x["id"]))
            print(f"\n\n\n***************\neval_list\n***************\n{json.dumps(eval_list)}")
            print(f"\n\n\n***************\nresults\n***************\n{results}")
        return results, eval_list
