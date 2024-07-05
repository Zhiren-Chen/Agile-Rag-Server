from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import List, Union
import json
import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 
from handler.disp import disp
import traceback
from configs import LIVECONFIGS_PATH, DEBOUNCE_TIME, MODIFY_TIMEOUT
import time
from utils import access_nested_dict

class Manager(FileSystemEventHandler):

    debounce_time: float = DEBOUNCE_TIME
    callback_factories: dict = {}
    keychains_hub: dict = {} #collects keychains for each callback
    simple_callbacks: dict = {}
    is_modifying: bool
    reset_dummy: Union[str, int, None]
    subors: list

    def __init__(self):
        self.last_config = json.load(open(LIVECONFIGS_PATH))
        self.last_call_time = time.time()
        self.is_modifying = False
        self.reset_dummy = None
        self.subors = ['llm','embedder','reranker']

    def add_simple_callback(self, name, callback):
        assert not name in self.simple_callbacks.keys()
        self.simple_callbacks[name] = callback

    def add_callback_factory(self, name, callback_factory, keychains):
        assert type(keychains[0]) == list
        if name in self.keychains_hub.keys():
            disp.red(f"Manager.add_callback_factory: name {name} already exists")
            raise ValueError(f'Manager.add_callback_factory(): name {name} already exists.')
        self.callback_factories[name] = callback_factory
        self.keychains_hub[name] = keychains

    def on_modified(self, event):
        if event.src_path.endswith('.json'):
            with open(event.src_path, 'r') as file:
                try:
                    new_config = json.load(file)
                except Exception as e:
                    disp.red("Config file JSONDecodeError. This update is cancelled. Try again later.")
                    print(e)
                    return
                try:
                    new_reset_dummy = new_config['reset_flags']
                    if self.reset_dummy is not None and not self.reset_dummy == new_reset_dummy:
                        disp.purple("reset all flags triggered")
                        for subor in self.subors:
                            self.simple_callbacks[f"reset_{subor}_flags"]()
                        self.simple_callbacks['reset_ordinator_flags']()
                        self.is_modifying = False
                    self.reset_dummy = new_reset_dummy
                    if self.is_modifying:
                        disp.yellow("conflicting modify queued")
                        for i in range(MODIFY_TIMEOUT):
                            time.sleep(1)
                            if not self.is_modifying:
                                break
                            if i%10 == 0:
                                disp.yellow("Been waiting for another ongoing modification for {i} seconds")
                    current_time = time.time()
                    if current_time - self.last_call_time < self.debounce_time:
                        disp.white("bouncing modify cancelled")
                        return
                    self.last_call_time = current_time
                    self.is_modifying = True
                    for name in self.keychains_hub.keys():
                        keychains = self.keychains_hub[name]
                        callback_factory = self.callback_factories[name]
                        new_concerned_configs = [access_nested_dict(new_config, keychain) for keychain in keychains]
                        old_concerned_configs = [access_nested_dict(self.last_config, keychain) for keychain in keychains]
                        concerned_comparisons = [new_concerned_configs[i] == old_concerned_configs[i] for i in range(len(new_concerned_configs))]
                        if False in concerned_comparisons:
                            disp.white(f"update starts for {name}.")
                            callback = callback_factory() #gets the callback function
                            callback(new_config,keychains)
                            disp.green(f"update successful for {name}.")
                except json.JSONDecodeError:
                    disp.red("Live update failed due to invalid json file, please double check the format")
                    new_config = self.last_config
                except Exception as e:
                    disp.red("Live update failed. 1) Make sure liveconfig.py is under the main directory with valid json format. 2) Check callback functions.")
                    traceback.print_exc()
                    new_config = self.last_config
            self.last_config = new_config
        self.is_modifying = False

manager = Manager()

observer = Observer()
observer.schedule(manager, path=LIVECONFIGS_PATH, recursive=False)
observer.start()
disp.blue("Watching for changes...")

#python handler/manager.py
if __name__ == "__main__":

    def make_update_callback(*args, **kwargs):
        def callback(configs,keychains):
            time.sleep(0.4)
            print("### manager.py make_update_callback() testargs:",args, kwargs)
        return callback

    manager.add_callback_factory(
        "test_callback",
        lambda: make_update_callback(time.time(), timenow = time.time()),
        [['embedder_params',],]
        )

    try:
        i=0
        while True:
            time.sleep(1)  
    except KeyboardInterrupt:
        observer.stop()
    observer.join()