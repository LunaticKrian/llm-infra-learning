from datasets import load_dataset
from torch.utils.data import Dataset


class MyDataset(Dataset):

    """
    定义初始化自定义数据集
    """
    def __init__(self, split):
        # 加载缓存数据集
        self.dataset = load_dataset(path="lansinuote/ChnSentiCorp",
                               cache_dir="/Users/krian/PycharmProjects/llm-study/02-Bert-Train/dataset/ChnSentiCorp")

        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        else:
            print("Invalid split")


    """
    获取数据集长度 
    """
    def __len__(self):
        return len(self.dataset)

    """
    获取数据
    """
    def __getitem__(self, item):
        # 获取数据集中的元素
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]

        return text, label


# 测试输出
if __name__ == '__main__':
    dataset = MyDataset("test")

    for text, label in dataset:
        print(text, label)