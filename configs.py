HOST="0.0.0.0"
PORT="8009"
USE_LLM = True
USE_EMBEDDER = True
USE_RERANKER = True
SERVICE_NUM_WORKERS = 12
TOTAL_CUDA_MEMORY = 8000 #MB, if you don't have shared VRAM, at least leave 20% margin. Otherwise, try not to include shared VRAM anyway.
MEMORY_REDUNDANCY = 0.15

#————————————————————CONFIGS RELATING TO liveconfigs.json, BE VERY CAREFUL———————————————————————————
LIVECONFIGS_PATH = 'liveconfigs.json'
DEBOUNCE_TIME = 0.4
MODIFY_TIMEOUT = 300
