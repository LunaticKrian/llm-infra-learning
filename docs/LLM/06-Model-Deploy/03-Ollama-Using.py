from openai import OpenAI

# 创基基于OpenAI API风格的API请求Client
client = OpenAI(base_url="http://localhost:11434/v1", api_key="<KEY>")

# 构建请求Prompt
chat_completion = client.chat.completions.create(messages=[
    {"role": "user", "content": "你好，请介绍一下自己"}
], model="qwen3:0.6b")

# 解析推理结果
response = chat_completion.choices[0]

print(response)
