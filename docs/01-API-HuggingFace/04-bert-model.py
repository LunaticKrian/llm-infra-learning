from transformers import BertTokenizer, BertForSequenceClassification, pipeline

model_dir = r"model/bert-base-chinese"

# 下载模型和分词器（如果配置了模型名，会先访问HuggingFace，然后判断本地目录是否存在模型文件，决定是否下载）
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", cache_dir=model_dir)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=model_dir)

# 创建pipline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cpu")

# 进行文件文本分类
result = classifier("你好，我是一个大语言模型")

# 输出模型分类结构
print(result)

# 输出模型结构
print(model)