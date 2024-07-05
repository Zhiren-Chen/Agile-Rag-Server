import os
import sys
from watchdog.observers import Observer
from typing import Any, List
import numpy as np
from scipy.interpolate import interp1d
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
from handler.disp import disp
import traceback
import time

def interp(vector, new_length):
        arr=np.array(vector)
        assert len(arr.shape)==1
        interpolated_arr = np.zeros(new_length)
        f = interp1d(np.arange(len(arr)), arr, kind='cubic')
        interpolated_arr = f(np.linspace(0, len(arr) - 1, new_length))
        return interpolated_arr.tolist()

class BasicEmbedder():
    
    models: dict
    standby_models: list[str] = []

    def __init__(self, models: dict, **kwargs):
        self.models = models
        for key, value in kwargs.items():
            setattr(self, key, value)

    def embed_documents(self, strings: List[str], model_name = None, output_dims: int = 0) -> List[List[float]]:
        if not self.models:
            disp.red("There is no available embedder to use.")
            return None, None
        embeddings = self.models.get(model_name,self.models[self.standby_models[0]])['model']
        vectors = embeddings.embed_documents(strings)
        if output_dims == 0:
            return vectors, embeddings.model_name
        else:
            return [interp(v, output_dims) for v in vectors], embeddings.model_name

    def embed_query(self, string: str, model_name = None, output_dims: int = -1) -> List[float]:
        vector = self.models.get(model_name,self.models[list(self.models.keys())[0]])['model'].embed_query(string)
        if output_dims == 0:
            return vector
        else:
            return interp(vector, output_dims)



    



