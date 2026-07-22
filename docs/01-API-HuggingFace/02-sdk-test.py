from transformers import AutoModelForCausalLM, AutoTokenizer

# 将模型和分词器下载到本地，并指定保存路径
model_name = "uer/gpt2-chinese-cluecorpussmall"

# 本地缓存路径
cache_dir = "/Users/krian/PycharmProjects/llm-study/01-API-HuggingFace/model/uer/gpt2-chinese-cluecorpussmall"

# 下载大模型（默认存储本地路径：～/.cache/huggingface/hub）
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
# 下载分词器
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型分词器已经下载到：{cache_dir}")