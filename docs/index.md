# LLM 基础设施学习笔记

欢迎来到 **大语言模型（LLM）基础设施** 学习笔记库。这里记录了从模型 API 调用、微调训练到部署落地的完整实践过程。

!!! tip "如何使用本站"
    - 顶部 **标签栏** 切换主题大类
    - 左侧 **侧边栏** 浏览具体章节
    - 右上角 🔍 **搜索** 快速定位内容
    - 右上角 🌗 **图标** 切换深 / 浅色模式

---

## 📚 内容导航

<div class="grid cards" markdown>

- :material-api: **HuggingFace**

    ---

    大模型 API 调用与 Transformers 库核心组件使用。

    [:octicons-arrow-right-24: 查看笔记](LLM/01-API-HuggingFace/note.md)

- :material-brain: **Bert**

    ---

    Bert 模型的微调训练与下游任务使用。

    [:octicons-arrow-right-24: 训练](LLM/02-Bert-Train/note.md) ·
    [:octicons-arrow-right-24: 使用](LLM/03-Bert-Using/note.md)

- :material-robot: **GPT**

    ---

    GPT 系列模型的训练流程与生成式使用。

    [:octicons-arrow-right-24: 训练](LLM/04-GPT-Train/note.md) ·
    [:octicons-arrow-right-24: 使用](LLM/05-GPT-Using/note.md)

- :material-server-network: **部署与微调**

    ---

    模型服务化部署与 LLaMA-Factory 高效微调。

    [:octicons-arrow-right-24: 部署](LLM/06-Model-Deploy/note.md) ·
    [:octicons-arrow-right-24: 微调](LLM/07-LLamaFactory-Finetune/note.md)

</div>

---

## 🚀 本地预览

```bash
pip install mkdocs-material      # 安装
mkdocs serve                     # 启动本地预览 → http://127.0.0.1:8000
mkdocs build                     # 构建静态站点到 site/
```
