from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

# 指定模型本地缓存路径（指定到config.json所在的文件目录的绝对路径）
model_dir = "/Users/krian/PycharmProjects/llm-study/01-API-HuggingFace/model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"

# 加载模型文件
model = AutoModelForCausalLM.from_pretrained(model_dir)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 配置pipeline，处理文本生成
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")

# 执行文本生成
outputs = generator("你好，我是一款大语言模型", max_length=256, num_return_sequences=1)

print(outputs)