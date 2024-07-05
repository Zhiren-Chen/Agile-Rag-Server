import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 
import torch
from fastapi.responses import StreamingResponse
from handler.manager import manager
from handler.disp import disp
from utils import get_remain_vram, decide_kick_idlers
import time
from configs import TOTAL_CUDA_MEMORY, MEMORY_REDUNDANCY

class Ordinator():

    is_reallocating: bool
    subors: list[str]
    callback_factories: dict
    
    def __init__(self):
        self.subors = ['embedder','reranker','llm']
        self.callback_factories = {}
        self.is_reallocating = False
        manager.add_simple_callback("reset_ordinator_flags", self.reset_flags)
    
    def add_callback_factory(self, name, callback_factory, **kwargs):
        self.callback_factories[name] = callback_factory

    def decide_upload_flex_device(self, model_class, model_dict):
        if model_dict.get("idle_on_cpu"):
            print("(Ordinator) a model with idle_on_cpu enabled will be uploaded to cpu")
            return 'cpu'
        i=0
        #while is_modifying or self.is_reallocating:
        while self.is_reallocating:
            time.sleep(0.2)
            i+=1
            if i%50 == 0:
                disp.yellow(f"Ordinator waiting for another modification for {i/5} seconds. Will default to cpu.")
                return 'cpu'
        self.is_reallocating = True
        remain_vram = get_remain_vram()
        print("(Ordinator) remain_vram:", remain_vram)
        if model_size := model_dict.get('size_est'):
            if remain_vram > model_size * (1+MEMORY_REDUNDANCY):
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = 'cpu'
        print("(Ordinator) Device decided:", device)
        self.is_reallocating = False
        return device

    def reallocate(self, model_class, model_name):
        #Analyzes the cuda memory usage of every model on cuda, then decides whether kick idle model(s) from cuda to cpu in order to release memory for another model
        target_model_info = self.callback_factories[f'get_{model_class}_models_info']()([model_name])
        #print("(Ordinator) reallocate() target_model_info:",target_model_info)
        if not target_model_info[model_name]['device'] == 'flex':
            return
        if target_model_info[model_name]['flex_device'] == 'cuda':
            return
        cuda_idlers = []
        for subor_model_class in self.subors:
            cuda_idlers += self.callback_factories.get(f'hold_{subor_model_class}_idlers', lambda: lambda: {})()()
        #print("(Ordinator) reallocate() cuda_idlers:",cuda_idlers)
        remain_vram = get_remain_vram()
        desired_vram = target_model_info[model_name]['size'] * (1+MEMORY_REDUNDANCY)
        print(f"(Ordinator) {remain_vram} remaining while {desired_vram} is needed for {model_class} {model_name}.")
        vram_to_release = desired_vram - remain_vram

        if vram_to_release < 0: 
            self.callback_factories[f'move_{model_class}_model']()(model_name, 'cuda')
            self.revive_holded_idlers(cuda_idlers)
            #NO NEED TO HOLD AND REVIVE TARGET MODEL becuase handled in livemodel
            #self.revive_holded_idlers([[model_class, model_name]])
            return 
        if len(cuda_idlers) == 0:
            #self.revive_holded_idlers([[model_class, model_name]]) #not needed
            disp.yellow(f"There is no flex model idling on cuda to be kicked. {model_name} will remain on cpu for infering.")
            return
        
        kick_choices = decide_kick_idlers(cuda_idlers, vram_to_release, desired_vram, )
        if not kick_choices:
            disp.yellow(f"Unable to find a satisfiable combination of cuda idlers to kick for freeing up {vram_to_release}MB, target model will remain on cpu.")
            self.revive_holded_idlers(cuda_idlers)
            return

        for choice in kick_choices:
            self.callback_factories.get(f"move_{choice[0]}_model")()(choice[1], 'cpu')
        #self.revive_holded_idlers([[model_class, model_name]])
        self.callback_factories.get(f"move_{model_class}_model")()(model_name, 'cuda')
        self.revive_holded_idlers(cuda_idlers)

    def revive_holded_idlers(self, cuda_idlers):
        for model_info in cuda_idlers:
            try:
                self.callback_factories[f"revive_{model_info[0]}_idlers"]()([model_info[1]])
            except:
                disp.red(f"failed to revive {model_info[0]} {model_info[1]}")
                import traceback
                traceback.print_exc()

    def reset_flags(self):
        self.is_reallocating = False

ordinator = Ordinator()
