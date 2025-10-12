from modelscope import snapshot_download

# 下载模型
model_dir = snapshot_download("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                              cache_dir="/Users/krian/PycharmProjects/llm-study/17-LlamaIndex-Demo/model")

