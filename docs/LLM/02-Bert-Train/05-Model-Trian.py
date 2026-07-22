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

import torch
from transformers import BertModel

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# 加载预训练模型
predict_model = BertModel.from_pretrained("/Users/krian/PycharmProjects/llm-study/02-Bert-Train/model/bert-base-chinese/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea")

print(predict_model)


# 定义下游任务（增量模型）
class Model(torch.nn.Module):

    """
    设计模型结构
    """
    def __init__(self):
        super().__init__()

        # 设计全链接网络，实现二分类任务
        self.fc = torch.nn.Linear(768, 2)


    """
    使用模型处理数据（执行前向计算）
    """
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 冻结Bert模型的参数，使其不参与训练
        with torch.no_grad():
            out = predict_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 增量模型参与训练
        out = self.fc(out.last_hidden_state[:, 0])

        return out



#模型训练
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer,AdamWeightDecay

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义训练的轮次(将整个数据集训练完一次为一轮)
EPOCH = 30000

#加载字典和分词器
token = BertTokenizer.from_pretrained(r"D:\PycharmProjects\demo_02\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

#将传入的字符串进行编码
def collate_fn(data):
    sents = [i[0]for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        # 当句子长度大于max_length(上限是model_max_length)时，截断
        truncation=True,
        max_length=512,
        # 一律补0到max_length
        padding="max_length",
        # 可取值为tf,pt,np,默认为list
        return_tensors="pt",
        # 返回序列长度
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(label)
    return input_ids,attention_mask,token_type_ids,label



#创建数据集
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    #训练批次
    batch_size=90,
    #打乱数据集
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    #对加载的数据进行编码
    collate_fn=collate_fn
)
if __name__ == '__main__':
    #开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    #定义优化器
    optimizer = AdamWeightDecay(model.parameters())
    #定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i,(input_ids,attention_mask,token_type_ids,label) in enumerate(train_loader):
            #将数据放到DVEVICE上面
            input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),label.to(DEVICE)
            #前向计算（将数据输入模型得到输出）
            out = model(input_ids,attention_mask,token_type_ids)
            #根据输出计算损失
            loss = loss_func(out,label)
            #根据误差优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #每隔5个批次输出训练信息
            if i%5 ==0:
                out = out.argmax(dim=1)
                #计算训练精度
                acc = (out==label).sum().item()/len(label)
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")
        #每训练完一轮，保存一次参数
        torch.save(model.state_dict(),f"params/{epoch}_bert.pth")
        print(epoch,"参数保存成功！")

