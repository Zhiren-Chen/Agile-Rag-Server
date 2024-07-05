## Agile Rag Server可以利用显存只够容纳一个LLM的CUDA显卡，运行LLM、向量化和重排等三个服务。
### 已成功在Langchain-Chatchat、FastGPT、Vanna等RAG应用上验证可行性，并在**单张RTX4060 Laptop (8GB)** 及32GB cpu内存的硬件配置下，实现了**20秒内**完成Qwen1.5-7B-Chat-GPTQ-Int4（需8GB显存）推理 -> bge-m3（需3GB内存）检索向量 -> bge-reranker-v2-m3（需3GB显存）重排**100段**最长768字的文本 -> Qwen1.5-7B-Chat-GPTQ-Int4再次推理回答的完整RAG流程，使得**中低端显卡也可以兼顾大模型推理和大规模重排**。
### 最佳适用场景：带有单张6G-12G显卡的个人电脑，且最好拥有24G以上的cpu内存。
#### 工作原理：按照各模型的使用情况，把它们在cuda和cpu之间调遣（一般耗时约1秒），把显存安排给大任务，使用cpu内存等待或执行小任务（如少量的向量化）。
#### 其他功能：除自动派遣模型外，在不中止服务的情况下，还支持对各模型参数的热修改、手动派遣device、模型热切换、删除和新模型的上传。(通过实时修改liveconfigs.json实现）
### 用户只需把LLM、向量模型和重排模型交给AgileRagServer统一管理，并在liveconfigs.json中合理设置各模型的配置参数。
### APIs: **支持OpenAI的chat和embeddings调用，其中chat支持流式输出，带有自定义的重排服务。**利用asyncio和concurrent实现异步并发推理，处理并发请求效率极高。服务基于FastAPI框架。
### 测试过的模型包括Qwen2-7B-Instruct-AWQ，Qwen1.5-7B-Chat-GPTQ-Int4，Qwen1.5-4B-Chat-AWQ，bge-m3，m3e-base，bce-reranker-base_v1，bge-reranker-v2-m3等。其他任何模型只要允许在cpu上临时存放和在cuda上推理，都应该可以在对代码稍加修改后使用。
## —————————————————————————————————————
## Agile Rag Server enables you to run LLM, embedder and reranker, all on CUDA, with very limited VRAM just enough to run the LLM. 
### Designed for: computers with single NVIDIA graphics with 6G-12G VRAM, along with more than 24G bgcpu memory.
### The benefits of AgileRagServer is proved on RAG applications including Langchain-Chatchat, FastGPT, and Vanna. **Within 20 seconds**, it successfully achieved a complete RAG cycle including Qwen1.5-7B-Chat-GPTQ-Int4（8GB VRAM）inference -> bge-m3（3GB RAM）vector search -> bge-reranker-v2-m3（3GB VRAM）reranking **100 text blocks** with max length of 768 -> Qwen1.5-7B-Chat-GPTQ-Int4 inferencing again, enabling mid-lower-tier GPUs to reconcile LLM inferencing and massive rerank.
#### Logic: dispatch each model to cpu or cuda according to usage (usually takes about 1s), assigns cuda memory to heavy tasks, and uses cpu for idling or smaller tasks (like minor embeddings).
#### Additional features: While the service is running, you can also edit model attributes, manually dispatch device, and switch/delete/upload models. (achieved by editting liveconfigs.json while running)
### The user only needs to let AgileRagServer to ordinate all LLMs, embedders and rerankers, and appropriately configure liveconfigs.json
### APIs: **OpenAI embeddings and chat (supports stream output), and custom rerank services.** Utilizes asyncio and concurrent to handle concurrent requests, providing very high efficiency. The service part is based on FastAPI.
### Tested models include Qwen2-7B-Instruct-AWQ，Qwen1.5-7B-Chat-GPTQ-Int4，Qwen1.5-4B-Chat-AWQ，bge-m3，m3e-base，bce-reranker-base_v1，and bge-reranker-v2-m3. With some modification to the code, other models should work fine as long as they allow idling on cpu and inferencing on cuda.
## ——————————————————————————————————————
## 使用指南
### 1. 找到configs.py并按情况填写，这里面的配置不支持热修改。
### 2. 填写liveconfigs.json，这里所有参数支持冷修改或热修改，每次改动之后，记得ctrl+s保存才会生效。以下是一个带解析的示例：
```
{
    "reset_flags": 0, //如果整个程序出bug卡住了，就改一下这里，不管改成啥，只要有改动（有时候需要连改两下）就会重置程序内部的所有线程互锁机制

    "standby_llms": ["qwen7b"], //待命的LLM，需是llms中的一个或几个（32G以下内存建议一个就好），客户端可按名称调用，调的没有就默认第一个
    "standby_embedders": ["bge-m3","m3e-base"], //待命的向量模型，需是下面embedders中的一个或多个，这里有的客户端都能调用，调的没有就默认第一个
    "standby_rerankers": ["bce","bge_v2m3"], //待命的重排模型，需是下面rerankers中的一个或多个，这里有的客户端都能调用，调的没有就默认第一个

    "llms":{ //各个大模型详细信息，放多少个都行，standby_llms里没有的就不会加载
        "qwen7b": { //名称随便取，跟standby_llms统一就行，客户端也需要按这个名字来调用
            "llm_path": "/home/usr/codes/models/qwen7b/qwen/Qwen2-7B-Instruct-AWQ", //路径，必填，按需修改。热修改将导致重新加载
            "tokenizer_path": "/home/usr/codes/models/qwen7b/qwen/Qwen2-7B-Instruct-AWQ", //路径，必填，按需修改。热修改将导致重新加载
            "device":"flex", //必填，flex、cuda或cpu，只有flex才允许模型被自动调遣，否则将固定在cuda或cpu上。重复一下，这个也是可以热修改的！
            "max_tokens": 2048, //模型允许长度，必填
            "size_est": 5800, //选填（就是指整行可以去掉），对模型大小（MB）的估测，
            "cuda_thres": 0, //选填，接收到的任务包含多少条数据时才考虑分配cuda，仅在device为flex时有效。0或不填就是一收到任务就申请cuda。
            "idle_on_cpu": false //选填，仅在device为flex时有效，意为每次任务完成后自觉退居cpu而不等被别的模型踢下来。
        },
        "qwen7bgptq": {
            "llm_path": "/home/usr/codes/models/models/qwen/Qwen1___5-7B-Chat-GPTQ-Int4",
            "tokenizer_path": "/home/usr/codes/models/models/qwen/Qwen1___5-7B-Chat-GPTQ-Int4",
            "device":"flex", //如果要把这里手动改成cuda，需特别注意先腾出足够显存！！（就是把某些cuda的模型手动改cpu，自己按需而定）
            "max_tokens": 2048,
            "size_est": 6500, //仅用于首次从硬盘加载时且device为flex时决定cuda还是cpu（不填就暂时传到cpu），然后真实大小就会在程序内部测出来。
            "cuda_thres": 0,
            "idle_on_cpu": false //某些情况下可节约1-2秒。对于大多数RAG任务，LLM建议false
        }
        //可添加更多模型
    },
    "llm_params":{ //LLM通用参数
        "default_temp": 0.5, //客户端不指定温度时用这个温度
        "force_0_temp": false, //不论客户端指定多少温度，都强制0
    },
    "embedders":{ //向量模型，跟上面一个道理
        "bge-m3":{ //一个道理，跟上面没区别就不解释
            "path": "/home/usr/codes/models/bge-m3", 
            "device": "flex", //对于大批量连发的请求（例如构建向量库），推荐手动设为cuda（需特别注意要手动把cuda显存腾出来），完事再热修改回flex
            "size_est": 2900,
            "cuda_thres": 20,
            "idle_on_cpu": false //一般的RAG任务非常不建议填true，因为embed请求一般是连发，填true会导致来回来去的在cpu和cuda之间横跳
        },
        "m3e-base": { //略
            "path": "/home/usr/codes/models/m3e-base", 
            "device": "flex",
            "size_est": 500,
            "cuda_thres": 100,
            "idle_on_cpu": false
        }
        //可添加更多模型
    },
    "embedder_params":{ //embedder通用参数
        "token_estimation": "len", // 文本总长度估测：len 或 re 。"len"直接调用len(str)返回总字符数，非常适合中文，但不适合英文。"re"则利用re分词后统计
        "resize": 0 //0为保持原本向量，设为正整数时将调用插值函数，用于强行匹配不同的embedding，不推荐使用
    },
    "rerankers":{
        "bce":{
            "path":"/home/usr/codes/models/bce-reranker-base_v1",
            "max_tokens": 512,
            "device": "flex",
            "cuda_thres": 10,
            "idle_on_cpu": true //对于大多数RAG任务，推荐使用true，因为重排一般不会连续调取两次，最好让它用完cuda就立马让出资源，可节约1-2秒
        },
        "bge_v2m3":{
            "path":"/home/usr/codes/models/bge_reranker/AI-ModelScope/bge-reranker-v2-m3",
            "max_tokens": 768,
            "device": "flex",
            "cuda_thres": 8,
            "idle_on_cpu": true,
            "size_est": 2800
        }
        //可添加更多模型
    },
    "reranker_params":{
        "top_n": 9999, //限制返回的分值数量（前n名），希望全部返回就设特别大的数，比如9999
        "sentence_transformer_num_workers": 0, //用0即可
        "batch_size": 32 //一般用32即可
    }
}
```
