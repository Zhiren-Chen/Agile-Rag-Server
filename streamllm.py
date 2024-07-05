import os
import sys
from typing import Any, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import json
import numpy
from basic.basicllm import BasicLLM
from basic.livemodel import AbsLiveModel
from utils import access_nested_dict, get_model_size
from handler.manager import manager, observer
from handler.disp import disp
from handler.ordinator import ordinator
import time
import torch
import gc 
from configs import LIVECONFIGS_PATH
import traceback
import threading
from datetime import datetime

class StreamLLM(BasicLLM, AbsLiveModel):
    
    def __init__(self,):
        super().__init__()
        params_keychains = [['llm_params',],]
        model_keychains = [['standby_llms',],['llms',]]
        configs = json.load(open(LIVECONFIGS_PATH, 'r'))
        ordinator.add_callback_factory(
            "get_llm_models_info",
            lambda: self.make_info_callback(tensorkeys = ['llm','tokenizer']))
        ordinator.add_callback_factory(
            "move_llm_model",
            lambda: self.make_move_callback())
        ordinator.add_callback_factory(
            "hold_llm_idlers",
            lambda: self.make_hold_idlers_callback(tensorkeys = ['llm','tokenizer']))
        ordinator.add_callback_factory(
            "revive_llm_idlers",
            lambda: self.make_revive_callback())
        self.make_update_params_callback()(configs, params_keychains)
        self.make_reload_model_callback(['llm','tokenizer'])(configs, model_keychains)
        manager.add_simple_callback("reset_llm_flags", self.reset_flags)
        manager.add_callback_factory(
            "llm_params",
            lambda: self.make_update_params_callback(), 
            params_keychains)
        manager.add_callback_factory(
            "llm_models",
            lambda: self.make_reload_model_callback(), 
            model_keychains)        

    def load_model(self, model_name: str, model_dict: dict,**kwargs):
        device = ordinator.decide_upload_flex_device('llm', model_dict) if model_dict['device'] == 'flex' else model_dict['device']
        try:
            llm = AutoModelForCausalLM.from_pretrained(model_dict["llm_path"],
                    device_map=device
                )
        except:
            disp.red(f"Having problem to load llm for {model_name}. Check the paths.")
            traceback.print_exc()
            return
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dict["tokenizer_path"])
        except:
            disp.red(f"Having problem to load llm and/or tokenizer for {model_name}. Check the paths.")
            traceback.print_exc()
            try:
                llm.to('cpu')
                del llm
            except:
                pass
            return
        self.models[model_name] = {
            'llm_path': model_dict['llm_path'],
            'tokenizer_path': model_dict['tokenizer_path'],
            'max_tokens': model_dict['max_tokens'],
            'size': get_model_size(llm.model),
            'llm': llm,
            'tokenizer': tokenizer,
            'device': model_dict['device'],
            'flex_device': device,
            "cuda_thres": model_dict.get('cuda_thres',0),
            "idle_on_cpu": model_dict.get('idle_on_cpu',False),
            'is_loading': False,
            'count_infering': 0
        }
        disp.blue(f"loaded LLM and tokenizer: {model_name}, with LLM size {self.models[model_name]['size']} MB")

    def unload_model(self, model_name: str, **kwargs):
        self.models[model_name]['llm'].to('cpu')
        del self.models[model_name]['llm']
        del self.models[model_name]['tokenizer']
        del self.models[model_name] 
        torch.cuda.empty_cache()
        gc.collect()
        disp.blue(f"deleted {model_name} from standby llms, remaining: {list(self.models.keys())}")

    def update_model(self, model_name: str, model_config: dict):
        if not self.models[model_name]['tokenizer_path'] == model_config['tokenizer_path'] or not self.models[model_name]['llm_path'] == model_config['llm_path']:
            self.unload_model(model_name)
            self.load_model(model_name, model_config)
            disp.purple(f"LLM model {model_name} reloaded")
        else:
            self.maybe_move_model(model_name, model_config['device'], model_config.get('idle_on_cpu'))
            self.models[model_name]['max_tokens'] = model_config['max_tokens']
            self.models[model_name]["cuda_thres"] = model_config.get('cuda_thres',0)
            self.models[model_name]["idle_on_cpu"] = model_config.get('idle_on_cpu',False)

    async def stream(self, 
            messages: dict,
            temperature: float = -1,
            max_new_tokens: int = -1,
            chunk_size: int = -1,
            verbose: bool = False,
            model_name: Optional[str] = None):
        if not self.models:
            disp.red("There is no available LLM to use.")
            raise ValueError("There is no available LLM to use.")
        if not (model_name is not None and model_name in self.models.keys()):
            disp.yellow(f"Requested {model_name} is not available, using {self.standby_models[0]} instead")
            model_name = self.standby_models[0] 
        self.apply_reallocate(model_name)
        self.models[model_name]['count_infering']+=1
        i=0
        while self.models[model_name]['is_loading']:
            if i%50 == 0:
                disp.yellow(f"Embedder embed_as_openai() waiting for loading {model_name} for {i*10}s")
            time.sleep(0.1)
            i+=1
        if not self.models[model_name]['flex_device'] == 'cuda':
            self.models[model_name]['count_infering']-=1
            disp.red(f"{model_name} is not on cuda. The LLM must be on cuda for inferencing.")
            raise ValueError(f"{model_name} is not on cuda. The LLM must be on cuda for inferencing.")
        try:
            response, _, _ = self.once(messages,
                                temperature,
                                max_new_tokens,
                                verbose,
                                model_name)

            stream_chunk_size = chunk_size if chunk_size>0 else self.params.get('stream_chunk_size',10)
        except:
            disp.red(f"Failed to generated with {model_name}.")
            traceback.print_exc()
            raise ValueError(f"Failed to generated with {model_name}.")
        self.models[model_name]['count_infering']-=1
        if self.models[model_name].get('idle_on_cpu') and self.models[model_name]['device'] == 'flex':
            if self.models[model_name]['count_infering']==0 and not self.models[model_name]['flex_device'] == 'cpu':
                move_thd = threading.Thread(target=self.move_model, args=(model_name, 'cpu'))
                move_thd.start()
        if verbose:
            print("### StreamLLM stream() response:",response)
            print("### StreamLLM stream() stream_chunk_size:",stream_chunk_size)

        for i in range(0, len(response), abs(stream_chunk_size)):
            resp = response[i:i+stream_chunk_size]
            finish_reason = 'stop' if i+stream_chunk_size>=len(response) else None
            yield resp, finish_reason

    async def stream_as_openai(self,
                                messages: dict, 
                                temperature: float = -1, 
                                max_new_tokens: int = -1,
                                chunk_size: int = -1,
                                created: str = '1999999999',
                                message_id: str = 'cmpl-1234567890abcdef1234567890abcdef',
                                verbose: bool = False,
                                model_name: Optional[str] = None):
        
        model_name_likely_used = model_name if model_name in self.models.keys() else list(self.models.keys())[0] if self.models else None
        async for resp, finish_reason in self.stream(messages,
                                            temperature,
                                            max_new_tokens,
                                            chunk_size,
                                            verbose,
                                            model_name):
            respdict = {
                "id":message_id,
                "object":"chat.completion.chunk",
                "created":created,
                "model": model_name_likely_used,
                "choices":
                [
                    {
                        "index":0,
                        "delta":{
                            "content":resp,
                            },
                        "logprobs":None,
                        "finish_reason": finish_reason,
                    }
                ]
            }
            yield f"data: {json.dumps(respdict)}\n\n"
        yield f"data: [DONE]\n\n"

llm = StreamLLM()    
        
#python streamllm.py
if __name__ == "__main__":
    prompt = "讲个笑话"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    async def respond():
        async for response in llm.stream_as_openai(messages, temperature = 0.2, max_new_tokens=150, chunk_size=10, verbose=False):
            print("generated response:",response)
    print("Stream output starts")
    asyncio.run(respond())
    print("Stream output ends")
    print('answer at once without stream:')
    ans = llm.once(messages, temperature = 2, max_new_tokens=150, verbose=False)
    print(ans)
    


        