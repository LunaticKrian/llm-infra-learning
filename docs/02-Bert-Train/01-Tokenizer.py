from transformers import BertTokenizer

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(
    "/Users/krian/PycharmProjects/llm-study/02-Bert-Train/model/bert-base-chinese/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea")

# 输出分词器结构
# print(tokenizer)

# 定义一个句子文本集合
sentences = [
    "白日依山尽",
    "价格在这个地方适中，附近有早餐店，比较方便，地理位置好"
]

# 批量进行句子解码
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sentences[0], sentences[1]],
    add_special_tokens=True,
    # 当句子的长度大于max_length时进行句子截断
    truncation=True,
    # 设置句子编码之后输出的序列的最大长度，
    # 上限是模型能处理的最大长度：model_max_length
    max_length=6,
    # 当句子的长度小于max_length时，一律补0
    padding="max_length",
    # 设置编码的返回类型（可取值：tf，pt，np，默认List（None））
    return_tensors=None,
    # 模型返回值
    return_attention_mask=True,
    return_token_type_ids=True,
    return_special_tokens_mask=True,
    # 返回编码后的序列长度
    return_length=True,
)

# input_ids：编码后序列
# token_type_ids：第一个句子和特殊符号的位置是0，第二句子的位置1（只针对上下文编码）
# special_tokens_mask：特殊符号位置为1，其他位置为0
# length：编码后的序列长度
for k, v in out.items():
    print(k, ": ", v)

# 解码文本
input_0 = tokenizer.decode(out["input_ids"][0])
input_1 = tokenizer.decode(out["input_ids"][1])

print(input_0, "\n", input_1)
