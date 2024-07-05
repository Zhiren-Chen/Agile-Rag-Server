from handler.disp import disp
from utils import access_nested_dict
import time
from abc import ABC, abstractmethod
from handler.manager import manager
from handler.ordinator import ordinator
import traceback
import torch
import gc

class AbsLiveModel(ABC):

    def make_update_params_callback(self, *args, **kwargs):
        def callback(configs: dict, keychains: list[list[str]]):
            assert len(keychains)==1 and type(keychains[0])==list
            params = [access_nested_dict(configs,keychains[0])][0]
            self.params = params
        return callback

    def make_reload_model_callback(self, tensorkeys = ['model',]):
        def callback(configs, keychains):
            assert len(keychains)==2 and type(keychains[0])==list
            params = [access_nested_dict(configs,keychain) for keychain in keychains]
            self.standby_models = params[0]
            models_info = {key: {k: v for k, v in sub_dict.items() if not k in tensorkeys} for key, sub_dict in self.models.items()} if len(list(self.models.keys()))>0 else {}
            models_should_load = {} 
            for name in params[0]:
                try:
                    models_should_load[name] = params[1][name]
                except:
                    disp.red(f"model {name} does not have associated information to load")
            #{"path":~,"device":~}
            #print("### AbsLiveModel make_reload_model_callback() models_should_load:",models_should_load)
            if "flex_device" in models_should_load.keys():
                disp.red("Do not include 'flex_device' as json keys, this is an internal attribute of the program.")
            for akey in list(self.models.keys()):
                if not akey in list(models_should_load.keys()):
                    i=0
                    while self.models[akey]['count_infering']>0 or self.models[akey]['is_loading']:
                        time.sleep(0.5)
                        i+=1
                        if i%20 == 0:
                            disp.yellow(f"AbsLiveModel make_reload_model_callback() callback() waiting for {akey} for {i}s (1)")
                    self.models[akey]['is_loading'] = True
                    self.unload_model(akey)
                    del models_info[akey]
                    #no need to change flag because deleted
                else:
                    self.update_model(akey, models_should_load[akey])
                
            for akey in models_should_load.keys():
                if not akey in models_info.keys():
                    self.load_model(akey, models_should_load[akey])
                    
        return callback

    def make_info_callback(self, tensorkeys = ['model',]):
        def callback(model_names = None):
            if not model_names:
                models_info = {key: {k: v for k, v in sub_dict.items() if not k in tensorkeys} for key, sub_dict in self.models.items()} if len(list(self.models.keys()))>0 else {}
                return models_info
            else:
                assert type(model_names) == list
                return {model_name: {k: v for k, v in self.models[model_name].items() if not k in tensorkeys} for model_name in model_names}
        return callback
    
    def make_move_callback(self):
        def callback(model_name, device_to):
            self.move_model(model_name, device_to)
            self.models[model_name]['flex_device'] = device_to
        return callback

    def make_hold_idlers_callback(self, tensorkeys=['model',]):
        holded_idlers = []
        def callback():
            for model_name in self.models.keys():
                if self.models[model_name]['device'] == 'flex':
                    if self.models[model_name]['flex_device'] == 'cuda':
                        if self.models[model_name]['count_infering'] == 0:
                            while True: #RISKY
                                if not self.models[model_name]['is_loading']:
                                    break
                            self.models[model_name]['is_loading'] = True
                            holded_idlers.append([self.model_class, model_name, self.models[model_name]['size']])
            return holded_idlers #returns a list of holded model_info dicts for non-infering models on cuda
        return callback

    def make_revive_callback(self):
        def callback(holded_names):
            #revive all models previously put on hold
            for model_name in holded_names:
                self.models[model_name]['is_loading'] = False
        return callback

    def reset_flags(self):
        for model_name in self.models.keys():
            self.models[model_name]['is_loading'] = False
            self.models[model_name]['count_infer'] = 0

    def move_model(self, model_name, device): 
        #this function should be covered by thread safe functions
        self.models[model_name]['is_loading'] = True
        while self.models[model_name]['count_infering']>0:
            time.sleep(0.5)
        try:
            self.models[model_name][self.tensorkey].to(device)
            self.models[model_name]['flex_device'] = device
            disp.blue(f"{self.model_class} {model_name} moved to {device}")
        except Exception as e:
            disp.red(f"failed to move {self.model_class} {model_name} to {device}")
            print(e)
            #traceback.print_exc()
        gc.collect()
        if not device == 'cuda':
            torch.cuda.empty_cache()
        self.models[model_name]['is_loading'] = False

    def maybe_move_model(self,model_name, device, idle_on_cpu = False):
        if not self.models[model_name]['device'] == device or not self.models[model_name]['idle_on_cpu'] == idle_on_cpu:
            if not device == 'flex' or (device == 'flex' and idle_on_cpu == True):
                i=0
                while self.models[model_name]['count_infering']>0 or self.models[model_name]['is_loading']:
                    time.sleep(0.5)
                    i+=1
                    if i%20 == 0:
                        print(self.models[model_name]['count_infering'], self.models[model_name]['is_loading'])
                        disp.yellow(f"AbsLiveModel maybe_move_model() waiting for {model_name} for {i}s (2)")
                self.models[model_name]['is_loading'] = True
                try:
                    self.move_model(model_name, device if not device == 'flex' else 'cpu')
                    self.models[model_name]['device'] = device
                except:
                    disp.red(f"AbsLiveModel maybe_move_model() failed to move {model_name}")
                self.models[model_name]['is_loading'] = False
            else:
                self.models[model_name]['device'] = device

    def apply_reallocate(self, model_name):
        while self.models[model_name]['is_loading']:
            time.sleep(0.2)
            #print("### apply_reallocate() waiting for ongoing loading")
        if self.models[model_name]['flex_device'] == 'cuda':
            return
        self.models[model_name]['is_loading'] = True
        i=0
        while self.models[model_name]['count_infering'] > 0 or ordinator.is_reallocating or manager.is_modifying:
            time.sleep(0.1)
            if i == 50: #if a model is infering while received the request to allocate, wait for maximum 5s 
                self.models[model_name]['is_loading'] = False
                disp.yellow(f"AbsLiveModel apply_reallocate() canceled relocate for {model_name} because it is infering.")
                return
        try:
            #print("### AbsLiveModel apply_reallocate() reallocating for ",model_name)
            ordinator.is_reallocating = True
            ordinator.reallocate(self.model_class,model_name)
            ordinator.is_reallocating = False
            #print("### AbsLiveModel apply_reallocate() finished reallocating for ",model_name)
        except:
            disp.red(f"reallocate for {model_name} failed")
            traceback.print_exc()
            ordinator.is_reallocating = False
        self.models[model_name]['is_loading'] = False


    @abstractmethod
    def load_model(self, model_name, model_dict,**kwargs):
        raise NotImplementedError('AbsLiveModel.load_model() must be implemented.')
    
    @abstractmethod
    def unload_model(self, model_name, **kwargs):
        raise NotImplementedError('AbsLiveModel.unload_model() must be implemented.')

    @abstractmethod
    def update_model(self, model_name, model_config):
        raise NotImplementedError('AbsLiveModel.update_model() must be implemented.')

    