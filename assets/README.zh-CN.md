# LLM 基础设施学习笔记

[🇬🇧 English](../README.md)

> 大语言模型（LLM）从 API 调用、微调训练到部署落地的完整学习笔记。

本项目记录了基于 HuggingFace / Transformers、Bert、GPT、LLaMA-Factory、LlamaIndex 等技术栈的实践过程，涵盖模型加载、训练微调、推理使用与服务化部署全流程。同时使用 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 将笔记构建为可搜索的静态文档站点。

---

## 📚 内容模块

| 模块 | 说明 |
| --- | --- |
| **HuggingFace** | 大模型 API 调用、Transformers 库核心组件使用 |
| **Bert 训练** | 中文评论分析模型的本地训练与测试 |
| **Bert 使用** | Bert 下游任务使用 |
| **GPT 训练** | GPT 模型训练流程（古文 / 歌词 / 对联等） |
| **GPT 使用** | Transformer 生成式模型使用 |
| **模型部署** | 模型服务化部署、Ollama 集成 |
| **LLaMA-Factory 微调** | LoRA 高效微调实践 |
| **LlamaIndex** | RAG 检索增强、Embedding 模型 |

> 每个模块目录下包含 `note.md`（笔记正文）与对应 `.py` 实现代码。

---

## 🚀 本地预览

本项目使用 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 作为文档引擎。

```bash
# 1. 安装（建议先创建虚拟环境）
python3 -m venv .venv && source .venv/bin/activate
pip install mkdocs-material

# 2. 启动本地预览（热重载）
mkdocs serve
# 浏览器打开 http://127.0.0.1:8000

# 3. 构建静态站点
mkdocs build      # 产物输出到 site/
```

---

## ☁️ 部署（Cloudflare）

站点通过 **Cloudflare** 自动构建部署，采用 Git 集成模式：

```
push 源码到 GitHub  →  Cloudflare 自动拉取  →  云端 mkdocs build  →  自动上线
```

关键配置：

| 配置项 | 值 |
| --- | --- |
| Framework preset | `MkDocs`（或 None） |
| Build command | `pip install mkdocs-material && mkdocs build` |
| Build output directory | `site` |

> 本地无需构建，也无需提交 `site/` 目录（已在 `.gitignore` 中忽略）。只提交 `mkdocs.yml` 和 `docs/` 源文件即可。

---

## 📁 项目结构

```text
.
├── mkdocs.yml            # 文档站点配置（主题、导航、扩展）
├── docs/                 # 笔记源文件（Markdown）
│   ├── index.md          # 站点首页
│   ├── stylesheets/      # 自定义样式
│   ├── LLM/              # LLM 各模块（API / Bert / GPT / 部署 等）
│   └── Pytorch/          # PyTorch 学习笔记
├── site/                 # 构建产物（gitignore，不提交）
└── README.md
```

---

## 📝 写作约定

- **新增笔记**：在 `docs/` 对应模块目录下创建 `*.md` 文件
- **出现在导航中**：导航由 `awesome-pages` 插件按目录树自动生成；在某目录下放置 `.pages` 文件即可控制标题与顺序
- Markdown 支持提示框（`!!! tip`）、标签页、代码复制、数学公式、Mermaid 图表等（详见 [Material 文档](https://squidfunk.github.io/mkdocs-material/reference/)）

---

## 🔗 相关链接

- [MkDocs Material 官方文档](https://squidfunk.github.io/mkdocs-material/)
- [MkDocs 文档](https://www.mkdocs.org/)
- [Cloudflare Pages 文档](https://developers.cloudflare.com/pages/)

---

> 🌐 语言：**中文** · [Switch to English](../README.md)
