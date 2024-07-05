import os
import sys
from typing import Any, List, Optional, Union
import numpy
from sentence_transformers import CrossEncoder
from configs import LIVECONFIGS_PATH
from basic.livemodel import AbsLiveModel
from handler.manager import manager, observer
from handler.disp import disp
from handler.ordinator import ordinator
import json
import time
import torch
import gc 
from utils import get_model_size
import threading
from datetime import datetime

class MovableCrossEncoder(CrossEncoder):

    def to(self, device):
        self._target_device = torch.device(device)
        self.model.to(device)

    @property
    def size(self):
        return get_model_size(self.model)

class Reranker(AbsLiveModel):
    
    params: dict
    models: dict
    model_class: str = 'reranker'
    tensorkey: str = 'model'
    standby_models: list[str] = []

    def __init__(self):
        self.params = {}
        self.models = {}
        self.model_class = 'reranker'
        params_keychains = [['reranker_params',],]
        model_keychains = [['standby_rerankers',],['rerankers',]]
        configs = json.load(open(LIVECONFIGS_PATH, 'r'))
        ordinator.add_callback_factory(
            "get_reranker_models_info",
            lambda: self.make_info_callback(tensorkeys = ['model',]))
        ordinator.add_callback_factory(
            "move_reranker_model",
            lambda: self.make_move_callback())
        ordinator.add_callback_factory(
            "hold_reranker_idlers",
            lambda: self.make_hold_idlers_callback())
        ordinator.add_callback_factory(
            "revive_reranker_idlers",
            lambda: self.make_revive_callback())
        self.make_update_params_callback()(configs, params_keychains)
        self.make_reload_model_callback()(configs, model_keychains)
        manager.add_simple_callback("reset_reranker_flags", self.reset_flags)
        manager.add_callback_factory(
            "reranker_params",
            lambda: self.make_update_params_callback(), 
            params_keychains)
        manager.add_callback_factory(
            "reranker_models",
            lambda: self.make_reload_model_callback(), 
            model_keychains)        
    
    def load_model(self, model_name: str, model_dict: dict, **kwargs):
        device = ordinator.decide_upload_flex_device('reranker', model_dict) if model_dict['device'] == 'flex' else model_dict['device']
        try:
            _model = MovableCrossEncoder(model_name=model_dict['path'], max_length=model_dict['max_tokens'], device=device)
        except Exception as e:
            disp.red(f"reranker {model_name} is not loaded. Make sure the path and configs are correct.")
            print(e)
            return
        self.models[model_name] = {
            "model": _model,
            "path": model_dict['path'],
            "device": model_dict['device'],
            "size": _model.size,
            "flex_device": device,
            "max_tokens": model_dict.get("max_tokens",512),
            "cuda_thres": model_dict.get('cuda_thres',0),
            "idle_on_cpu": model_dict.get('idle_on_cpu',False),
            "is_loading": False,
            "count_infering": 0
        }
        disp.blue(f"loaded reranker model {model_name} with size {self.models[model_name]['size']} MB")

    def unload_model(self, model_name: str, **kwargs):
        is_cuda = True if self.models[model_name].get('flex_device') == "cuda" else False
        self.models[model_name]['model'].to('cpu')
        del self.models[model_name]['model'] 
        del self.models[model_name] 
        if is_cuda:
            torch.cuda.empty_cache()
        gc.collect()
        disp.blue(f"deleted {model_name} from standby rerank models, remaining: {list(self.models.keys())}")

    def update_model(self, model_name: str, model_config: dict):
        if not self.models[model_name]['path'] == model_config['path']:
            self.unload_model(model_name)
            self.load_model(model_name, model_config)
            disp.purple(f"reranker model {model_name} reloaded")
        else:
            self.maybe_move_model(model_name, model_config['device'],model_config.get('idle_on_cpu'))
            self.models[model_name]["max_tokens"] = model_config.get("max_tokens",512)
            self.models[model_name]["cuda_thres"] = model_config.get('cuda_thres',0)
            self.models[model_name]["idle_on_cpu"] = model_config.get('idle_on_cpu',False)

    def rerank(
            self,
            documents: List[str],
            query: str,
            top_n: int = -1,
            model_name: Optional[Union[str, None]] = None
    ) -> dict:
        if not self.models:
            disp.red("Reranker.rerank() has no available model to use")
            return []
        if not (model_name is not None and model_name in self.models.keys()):
            disp.yellow(f"Requested {model_name} is not available, using {self.standby_models[0]} instead")
            model_name = self.standby_models[0] 
        if len(documents)>=self.models[model_name].get('cuda_thres',0) and self.models[model_name]['device'] == 'flex':
            self.apply_reallocate(model_name)
        self.models[model_name]['count_infering']+=1
        i=0
        while self.models[model_name]['is_loading']:
            if i%50 == 0:
                disp.yellow(f"Embedder embed_as_openai() waiting for loading {model_name} for {i*10}s")
            time.sleep(0.1)
            i+=1
        try:
            _model = self.models[model_name]['model']
            print(f"{datetime.now()} reranker using model {model_name} on {self.models[model_name]['flex_device']} for {len(documents)} pairs")
            if len(documents) == 0:  
                return []
            top_n = self.params['top_n'] if top_n == -1 else top_n
            sentence_pairs = [[query, _doc] for _doc in documents]
            results = _model.predict(sentences=sentence_pairs,
                                        batch_size=self.params['batch_size'],
                                        num_workers=self.params['sentence_transformer_num_workers'],
                                        convert_to_tensor=True #to avoid JSONDecodeError caused by numpy int64
                                        )
        except:
            import traceback
            disp.red(f"Failed to rerank {len(documents)} documents with {model_name}")
            traceback.print_exc()
        self.models[model_name]['count_infering'] -= 1
        if self.models[model_name].get('idle_on_cpu') and self.models[model_name]['device'] == 'flex':
            if self.models[model_name]['count_infering']==0 and not self.models[model_name]['flex_device'] == 'cpu':
                move_thd = threading.Thread(target=self.move_model, args=(model_name, 'cpu'))
                move_thd.start()
        del sentence_pairs
        rank = numpy.argsort(results.cpu()).tolist()[::-1]
        results_ordered = results.cpu().numpy()[rank].tolist()[:top_n]
        del results
        ranked_dicts = [{"index": rank[i], "relevance_score": results_ordered[i]} for i in range(len(results_ordered))]
        return ranked_dicts

    def get_top_n(
        self,
        documents: List[str],
        query: str,
        top_n: int = -1
    ) -> List[int]:
        if not self.models:
            disp.error("Reranker.rerank() has no available model to use")
            return []
        if not self.models:
            disp.red("Reranker has no available model to use")
            return {}
        if not (model_name is not None and model_name in self.models.keys()):
            disp.yellow(f"Requested {model_name} is not available, using {self.standby_models[0]} instead")
            model_name = self.standby_models[0]
        if len(documents)>=self.models[model_name].get('cuda_thres',0) and self.models[model_name]['device'] == 'flex':
            self.apply_reallocate(model_name)
        self.models[model_name]['count_infering']+=1
        i=0
        while self.models[model_name]['is_loading']:
            if i%50 == 0:
                disp.yellow(f"Embedder embed_as_openai() waiting for loading {model_name} for {i*10}s")
            time.sleep(0.1)
            i+=1
        try:
            _model = self.models[model_name]['model']
            print(f"{datetime.now()} reranker using model {model_name} on {self.models[model_name]['flex_device']} for {len(documents)} pairs")
            if len(documents) == 0:  
                return []
            top_n = self.params['top_n'] if top_n == -1 else top_n
            sentence_pairs = [[query, _doc] for _doc in documents]
            results = _model.predict(sentences=sentence_pairs,
                                        batch_size=self.params['batch_size'],
                                        num_workers=self.params['sentence_transformer_num_workers'],
                                        convert_to_tensor=True #to avoid JSONDecodeError caused by numpy int64
                                        )
        except:
            import traceback
            disp.red(f"Failed to rerank {len(documents)} documents with {model_name}")
            traceback.print_exc()
        self.models[model_name]['count_infering']-=1
        if self.models[model_name].get('idle_on_cpu') and self.models[model_name]['device'] == 'flex':
            if self.models[model_name]['count_infering']==0 and not self.models[model_name]['flex_device'] == 'cpu':
                move_thd = threading.Thread(target=self.move_model, args=(model_name, 'cpu'))
                move_thd.start()
        del sentence_pairs
        rank = numpy.argsort(results).cpu().tolist()[::-1]
        del results
        if top_n<0:
            top_n = RERANKER_TOP_N
        return rank[:top_n]

reranker = Reranker()

#python reranker.py
if __name__ == "__main__":

    query='一日三餐收费'
    documents = ["伙食补助费是指对工作人员出差期间给予的伙食补助费用。",
                    "不得提供高档酒水，白酒每x00毫升、红酒每x50毫升售价不得高于x00元。",
                    "从当日晚m时至次日晨n时连续乘车x小时以上的，可凭车票加发x0元伙食补助费。",
                    "对于单据不齐全、不清晰、不真实的，一律不得报销。",
                    "用餐标准\n早餐：a元；中餐：b元；晚餐：c元。（注意：每餐刷第二次起餐费翻倍）",
                    "严格控制陪餐人数\n1.公务接待陪餐人数：接待对象在x人以内的，陪餐人数不得超过y人",
                    "用知识库查找小米su7汽车的续航里程和电池容量，计算它的每公里耗电量"
    ]
    
    results = reranker.rerank(documents, query)
    print("results:", results)

