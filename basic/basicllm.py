import os
import sys
from typing import Any, List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
import time
from handler.disp import disp
import traceback
import threading
from datetime import datetime

class BasicLLM():

    models: dict
    params: dict
    model_class: str = 'llm'
    tensorkey: str = 'llm'
    standby_models: list[str] = []

    def __init__(self,
                 models: dict = {},
                 params: dict = {},
                 ):
        self.models = models
        self.params = params
        self.model_class = 'llm'
    
    def once(self, 
            messages: dict,
            temperature: float = -1,
            max_new_tokens: int = -1,
            verbose = False,
            model_name: Optional[str] = None):
        if not self.models: 
            disp.red("There is no LLM available to use.")
            return [], 0, 0
        model_dict = self.models.get(model_name,self.models[list(self.models.keys())[0]])
        temperature = self.params['default_temp'] if temperature == -1 else temperature
        temperature = 0 if self.params['force_0_temp'] else temperature
        max_new_tokens = model_dict['max_tokens'] if max_new_tokens < 0 else max_new_tokens
        print(f"{datetime.now()} LLM using model {model_name} on {self.models[model_name]['flex_device']}")
        text = model_dict['tokenizer'].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            temperature=temperature
        )
        if verbose:
            print("### temperature:",temperature)
            print("### max_new_tokens:",max_new_tokens)
            print("### chatmodels.py text (templated):",text)
        model_inputs = model_dict['tokenizer']([text], return_tensors="pt").to(model_dict['flex_device'])
        del text
        try:
            try:
                generated_ids = model_dict['llm'].generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
            except ValueError:
                if temperature<=0:
                    temperature=0.01
                elif temperature>=1:
                    temperature=0.99
                disp.yellow("WARNING: temperature clipped to",temperature)
                generated_ids = model_dict['llm'].generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature 
                )
        except RuntimeError:
            del model_inputs
            disp.red("Error encountered while llm making prediction.")
            traceback.print_exc()
            return '', 0, 0
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        input_len = len(model_inputs['input_ids'][0])
        del model_inputs
        response = model_dict['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_len = len(generated_ids[0])
        del generated_ids
        torch.cuda.empty_cache()
        if verbose:
            print("### chatmodels.py response:",response)
        return response, input_len, output_len

    def once_as_openai(self,
                        messages: dict,
                        temperature: float = -1, 
                        max_new_tokens: int = -1,
                        created: str = '1899999999',
                        message_id: str = 'cmpl-1234567891abcdef1234567890abcdef',
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
            response, input_len, output_len = self.once(
                messages,
                temperature,
                max_new_tokens,
                verbose,
                model_name,
            )
        except:
            traceback.print_exc()
        self.models[model_name]['count_infering'] -= 1
        if self.models[model_name].get('idle_on_cpu') and self.models[model_name]['device'] == 'flex':
            if self.models[model_name]['count_infering']==0 and not self.models[model_name]['flex_device'] == 'cpu':
                move_thd = threading.Thread(target=self.move_model, args=(model_name, 'cpu'))
                move_thd.start()
        retdict = { 'id': message_id,
                    'object': 'chat.completion',
                    'created': created,
                    'model': model_name if model_name in self.models.keys() else list(self.models.keys())[0] if self.models else None,
                    'choices': [{
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': response,
                            'tool_calls': []
                            },
                        'logprobs': None,
                        'finish_reason': 'stop',
                        'stop_reason': None}],
                        'usage': {
                            'prompt_tokens': input_len, 
                            'total_tokens': input_len+output_len, 
                            'completion_tokens': output_len}
                    }
        return retdict

        