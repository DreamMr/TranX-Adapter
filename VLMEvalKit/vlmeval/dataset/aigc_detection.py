from datetime import date
import re
import warnings

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import pandas as pd
from sklearn import metrics
import numpy as np
from collections import defaultdict


class ImageAIGCDetectionDataset(ImageBaseDataset):
    TYPE="AIGI"

    DATASET_URL = {
        "Chamelon": "https://huggingface.co/datasets/DreamMr/TranXAdapter-Dataset/blob/main/evaluation/Chamelon.tsv",
        "GenImage": "https://huggingface.co/datasets/DreamMr/TranXAdapter-Dataset/blob/main/evaluation/GenImage.tsv",
        "RR": "https://huggingface.co/datasets/DreamMr/TranXAdapter-Dataset/blob/main/evaluation/RR.tsv",
    }

    DATASET_MD5 = {
        "Chamelon": "93111652a59f915fe81c43c2d27df703",
        "GenImage": "c853cbd5e78a6d12249194f230fd73cc",
        "RR": "7492da6f5311573ae54ddea7e32e10ce",
    }


    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        prompt = ''
        prompt += f'Question: {question}\n'
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.aigi_eval import eval_results, report_metrics
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
        suffix = eval_file.split('.')[-1]
        nproc = judge_kwargs.pop('nproc', 32)
        nproc = 64
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}

        model = build_judge(**judge_kwargs)
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = eval_results(model, data, nproc, result_file)
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        # cal metrics
        dic = defaultdict(list)

        metrics_dict = report_metrics(data)
        dic['type'].append("Mean")
        for metric_name, metric_value in metrics_dict.items():
            dic[metric_name].append(metric_value)
        
        if 'type' in data.columns:
            type_set = set(data['type'].tolist())
            for tn in type_set:
                sub_data = data[data['type'] == tn]
                sub_metrics_dict = report_metrics(sub_data)
                dic['type'].append(tn)
                for metric_name, metric_value in sub_metrics_dict.items():
                    dic[metric_name].append(metric_value)

        dic = pd.DataFrame(dic)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(dic, score_file)

        return dic

