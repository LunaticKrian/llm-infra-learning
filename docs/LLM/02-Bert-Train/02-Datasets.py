from datasets import load_dataset, load_from_disk

# 在线加载数据集
dataset = load_dataset(path="lansinuote/ChnSentiCorp",
                       cache_dir="/Users/krian/PycharmProjects/llm-study/02-Bert-Train/dataset/ChnSentiCorp")

print(dataset)

# 将数据集转换为CSV格式数据
dataset.to_csv(path_or_buf="/Users/krian/PycharmProjects/llm-study/02-Bert-Train/dataset")

# 从本地缓存中加载数据
# dataset = load_from_disk("指定本地数据存储路径")
#
# print(dataset)

# 输出数据集内容
train_data = dataset["train"]
for data in train_data:
    print(data)

# 加载CSV格式数据
# dataset = load_dataset(path="csv", data_files="指定CSV文件存储绝对路径")