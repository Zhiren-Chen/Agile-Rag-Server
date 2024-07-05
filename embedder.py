import os
import sys
from typing import Any, List, Dict
from configs import LIVECONFIGS_PATH
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from basic.basicembedder import BasicEmbedder
from basic.livemodel import AbsLiveModel
from utils import access_nested_dict, get_model_size
from handler.manager import manager, observer
from handler.ordinator import ordinator
from handler.disp import disp
import time
import re
import json
import torch
import gc 
import traceback
import threading
from datetime import datetime

def estimate_tokens(sentence : str) -> int: #only for estimate number of tokens
    chn_pattern = r'[\u4e00-\u9fa5\s.,?!+]+'
    eng_pattern = r'[^\u4e00-\u9fa5\s.,?!+]'
    remain_eng = re.sub(chn_pattern, ' ', sentence).strip()
    remain_chn = re.sub(eng_pattern, '', sentence).strip()
    remain_chn = re.sub(r'\s+', '', remain_chn)
    len_remain_eng = 0 if remain_eng.strip()=='' else len(remain_eng.strip().split(' '))
    return len_remain_eng+len(remain_chn.strip())

class MovableHuggingFaceEmbeddings(HuggingFaceEmbeddings):

    def to(self, device: str):
        self.client.to(device)
        self.client._target_device = torch.device(device)
        self.model_kwargs['device'] = device

    @property
    def size(self):
        return get_model_size(self.client)

class Embedder(BasicEmbedder,AbsLiveModel):

    params: dict
    model_class: str = 'embedder'
    tensorkey: str = 'model'

    def __init__(self,models,**kwargs):
        super().__init__(models, **kwargs)
        self.model_class = 'embedder'
        params_keychains = [['embedder_params',],]
        model_keychains = [['standby_embedders',],['embedders',]]
        configs = json.load(open(LIVECONFIGS_PATH, 'r'))
        ordinator.add_callback_factory(
            "get_embedder_models_info",
            lambda: self.make_info_callback(tensorkeys = ['model',]))
        ordinator.add_callback_factory(
            "move_embedder_model",
            lambda: self.make_move_callback())
        ordinator.add_callback_factory(
            "hold_embedder_idlers",
            lambda: self.make_hold_idlers_callback())
        ordinator.add_callback_factory(
            "revive_embedder_idlers",
            lambda: self.make_revive_callback())
        self.make_update_params_callback()(configs, params_keychains)
        self.make_reload_model_callback()(configs, model_keychains)
        manager.add_simple_callback("reset_embedder_flags", self.reset_flags)
        manager.add_callback_factory(
            "embedder_params",
            lambda: self.make_update_params_callback(), 
            params_keychains)
        manager.add_callback_factory(
            "embed_models",
            lambda: self.make_reload_model_callback(), 
            model_keychains)  

    def embed_as_openai(self, strings: List[str], model_name = None, output_dims: int = -1, tokens_estimation: str = 'len') -> Dict:
        if not self.models:
            disp.red("Embedder.embed_as_openai() has no available model to use")
            return {}
        if not (model_name is not None and model_name in self.models.keys()):
            disp.yellow(f"Requested {model_name} is not available, using {self.standby_models[0]} instead")
            model_name = self.standby_models[0] #!!!will throw exception if no model to use, should catch in servicesn (update: already defended)
        if len(strings)>=self.models[model_name].get('cuda_thres',0) and self.models[model_name]['device'] == 'flex':
            self.apply_reallocate(model_name)
        self.models[model_name]['count_infering']+=1
        i=0
        while self.models[model_name]['is_loading']:
            if i%50 == 0:
                disp.yellow(f"Embedder embed_as_openai() waiting for loading {model_name} for {i*10}s")
            time.sleep(0.1)
            i+=1
        try:
            print(f"{datetime.now()} embedder using model {model_name} on {self.models[model_name]['flex_device']} for {len(strings)} strings")
            output_dims = self.params['resize'] if output_dims==-1 else output_dims
            embeds, used_model_path = self.embed_documents(strings, model_name, output_dims)
            embedded_data=[
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embeds[i],
                    "input": strings[i]
                }
            for i in range(len(strings))]   
        except:
            traceback.print_exc()     
        self.models[model_name]['count_infering']-=1
        if self.models[model_name].get('idle_on_cpu')  and self.models[model_name]['device'] == 'flex':
            if self.models[model_name]['count_infering']==0 and not self.models[model_name]['flex_device'] == 'cpu':
                move_thd = threading.Thread(target=self.move_model, args=(model_name, 'cpu'))
                move_thd.start()
        tokens_estimation = self.params['token_estimation'] if tokens_estimation is None or tokens_estimation=='' else tokens_estimation
        if tokens_estimation == 're':
            #total_tokens = sum(list(map(lambda string: estimate_tokens(string), strings)))
            total_tokens = sum([estimate_tokens(input_string) for input_string in strings])
        else:
            total_tokens = sum([len(string) for string in strings])
        retdict={
            "object": 'embedding',
            "data": embedded_data,
            "model": used_model_path,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        return retdict

    def load_model(self, model_name: str, model_dict: dict,**kwargs):
        device = ordinator.decide_upload_flex_device('embedder', model_dict) if model_dict['device'] == 'flex' else model_dict['device']
        try:
            embeddings = MovableHuggingFaceEmbeddings(
                model_name=model_dict['path'],
                model_kwargs={**kwargs,**{'device': device}})
        except:
            disp.red(f"Having problem to load embeddings for {model_name}. Check the path and configs.")
            traceback.print_exc()
            return
        self.models[model_name] = {
            "model": embeddings,
            "path": model_dict['path'],
            "device": model_dict['device'],
            "flex_device": device,
            "size": embeddings.size,
            "cuda_thres": model_dict.get('cuda_thres',0),
            "idle_on_cpu": model_dict.get('idle_on_cpu',False),
            
            "count_infering": 0,
            "is_loading": False
        }
        disp.blue(f"loaded embedder model {model_name} with size {self.models[model_name]['size']} MB")

    def unload_model(self, model_name: str, **kwargs):
        is_cuda = True if self.models[model_name].get('flex_device') == "cuda" else False
        self.models[model_name]['model'].to('cpu')
        del self.models[model_name]['model'] 
        del self.models[model_name] 
        if is_cuda:
            torch.cuda.empty_cache()
        gc.collect()
        disp.blue(f"deleted {model_name} from standby embedder models, remaining: {list(self.models.keys())}")

    def update_model(self, model_name: str, model_config: dict):
        if not self.models[model_name]['path'] == model_config['path']:
            self.unload_model(model_name)
            self.load_model(model_name, model_config)
            disp.purple(f"embedder model {model_name} reloaded")
        else:
            self.maybe_move_model(model_name, model_config['device'], model_config.get('idle_on_cpu'))
            self.models[model_name]["cuda_thres"] = model_config.get('cuda_thres',0)
            self.models[model_name]["idle_on_cpu"] = model_config.get('idle_on_cpu',False)

    @classmethod
    def get_HF_embedder(cls,**kwargs):
        instance = cls(
                    models={},
                    **kwargs)
        return instance

embedder = Embedder.get_HF_embedder()

#python embedder.py
if __name__ == "__main__":
    emb = embedder.embed_as_openai(['你好','hello world'],None, 8, 're')
    print(f"output embeddings:",emb)
