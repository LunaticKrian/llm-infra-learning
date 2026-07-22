from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM


# 加载本地模型
llm = HuggingFaceLLM(model_name="/Users/krian/PycharmProjects/llm-study/06-Model-Deploy/model/Qwen/Qwen3-0.6B",
                     tokenizer_name="/Users/krian/PycharmProjects/llm-study/06-Model-Deploy/model/Qwen/Qwen3-0.6B",
                     model_kwargs={
                         "trust_remote_code": True
                     }, tokenizer_kwargs={
        "trust_remote_code": True
    })

# 请求模型进行推理生成
response = llm.chat(messages=[ChatMessage("Hello!")])

# 输出响应结果信息
print(response)
