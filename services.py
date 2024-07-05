from fastapi import FastAPI
import numpy as np
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from handler.disp import disp
from configs import *
import time

app = FastAPI()
server = uvicorn.Server(uvicorn.Config(app))

formatted_today = datetime.today().strftime('%Y%m%d')
service_count = 0

executor = ThreadPoolExecutor(max_workers=SERVICE_NUM_WORKERS)

if USE_LLM:
    from streamllm import llm
    from fastapi.responses import StreamingResponse

if USE_EMBEDDER:
    from embedder import embedder

if USE_RERANKER:
    from reranker import reranker

def get_openai_id(prefix = ''):
    global service_count
    service_count += 1
    if service_count == 99999999:
        service_count = 0
    return prefix + formatted_today + '1234567890abcdef' + str(service_count).zfill(8)

def get_openai_created():
    now = datetime.now()
    formatted_now = now.strftime('%Y%m%d%H%M')
    return formatted_now[-10:]

#——————————————————————————————————————向量化——————————————————————————————————————————

if USE_EMBEDDER:

    @app.post("/embeddings")
    async def embed(data: dict): 
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, embedder.embed_as_openai, data['input'], data.get('model'))
        return result

    #Alternative post
    @app.post("/v1/embeddings")
    async def embed2(data: dict):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, embedder.embed_as_openai, data['input'], data.get('model'))
        return result

#—————————————————————————————————————LLM服务———————————————————————————————————————————
#python main.py

if USE_LLM:
    
    @app.post("/v1/chat/completions")
    async def completions(data: dict):
        input_model = data.get('model')
        input_temp = data.get('temperature',-1)
        input_stream = data.get('stream',False)
        if not input_stream:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(executor, 
                                            llm.once_as_openai, 
                                            data['messages'], 
                                            input_temp,
                                            -1,
                                            get_openai_created(),
                                            get_openai_id('cmpl-'),
                                            False,
                                            input_model)
            return result
        else:
            return StreamingResponse(
                llm.stream_as_openai(data['messages'], 
                                    input_temp, 
                                    -1,
                                    -1,
                                    created = get_openai_created(),
                                    message_id = get_openai_id('cmpl-'),
                                    model_name = input_model
                                    ), 
                media_type="text/event-stream")

#———————————————————————————custome rerank functions, tested on FastGPT, non-related to OpenAI————————————————————————————————

if USE_RERANKER:
    
    @app.post("/v1/rerank")
    async def rerank(data: dict):
        top_n = data.get("top_n",data.get("n",9999))
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(executor, reranker.rerank, data['documents'], data['query'],top_n, data.get('model'))
        return {"results": results}

    @app.post("/rerank")
    async def rerank2(data: dict):
        top_n = data.get("top_n",data.get("n",9999))
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(executor, reranker.rerank, data['documents'], data['query'],top_n, data.get('model'))
        return {"results": results}

    @app.post("/v1/rerank/top_n")
    async def rerank_topn(data: dict):
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(executor, reranker.get_top_n, data['documents'], data['query'], data['top_n'], data.get('model'))
        return {"indices": json.dumps(results)}

#————————————————————————————————————————————————————————————————————————————————————————————————————————

@app.on_event("shutdown")
async def shutdown():
    server.should_exit = True
    disp.purple("stopping...")
    await asyncio.sleep(1)
