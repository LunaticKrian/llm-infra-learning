# LLM Infrastructure Learning Notes

[🇨🇳 中文版](./assets/README.zh-CN.md)

> End-to-end notes on Large Language Models (LLMs) — from API calls, fine-tuning, and training all the way to deployment.

This repository documents hands-on practice across the HuggingFace / Transformers, BERT, GPT, LLaMA-Factory, and LlamaIndex stacks, covering the full lifecycle: model loading, training & fine-tuning, inference, and serving. Notes are also built into a searchable static documentation site using [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

---

## 📚 Content Modules

| Module | Description |
| --- | --- |
| **HuggingFace** | LLM API calls and core components of the Transformers library |
| **BERT Training** | Local training & testing of a Chinese-review-analysis model |
| **BERT Usage** | Downstream tasks with BERT |
| **GPT Training** | GPT model training pipelines (classical Chinese / lyrics / couplets, etc.) |
| **GPT Usage** | Working with Transformer-based generative models |
| **Model Deployment** | Model serving & Ollama integration |
| **LLaMA-Factory Fine-tuning** | Efficient LoRA fine-tuning in practice |
| **LlamaIndex** | RAG retrieval-augmented generation & embedding models |

> Each module folder contains a `note.md` (the write-up) along with the corresponding `.py` implementation code.

---

## 🚀 Local Preview

This project uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) as its documentation engine.

```bash
# 1. Install (create a virtual environment first)
python3 -m venv .venv && source .venv/bin/activate
pip install mkdocs-material

# 2. Start the local preview server (hot reload)
mkdocs serve
# Open http://127.0.0.1:8000 in your browser

# 3. Build the static site
mkdocs build      # Output is written to site/
```

---

## ☁️ Deployment (Cloudflare)

The site is built and deployed automatically via **Cloudflare**, using Git integration:

```
push to GitHub  →  Cloudflare pulls source  →  cloud-side mkdocs build  →  auto-publish
```

Key configuration:

| Setting | Value |
| --- | --- |
| Framework preset | `MkDocs` (or None) |
| Build command | `pip install mkdocs-material && mkdocs build` |
| Build output directory | `site` |

> You never need to build locally or commit the `site/` directory (already in `.gitignore`). Just commit `mkdocs.yml` and the `docs/` source files.

---

## 📁 Project Structure

```text
.
├── mkdocs.yml            # Site config (theme, nav, extensions)
├── docs/                 # Note sources (Markdown)
│   ├── index.md          # Site home
│   ├── stylesheets/      # Custom stylesheets
│   ├── LLM/              # LLM modules (API / Bert / GPT / Deploy / ...)
│   └── Pytorch/          # PyTorch learning notes
├── site/                 # Build output (gitignored, not committed)
└── README.md
```

---

## 📝 Writing Conventions

- **Add a note:** create a `*.md` file under the relevant module folder in `docs/`
- **Make it appear in navigation:** navigation is auto-generated from the directory tree via the `awesome-pages` plugin; drop a `.pages` file in a folder to set its title and order
- Markdown supports admonitions (`!!! tip`), tabs, code copy, math, Mermaid diagrams, and more — see the [Material reference](https://squidfunk.github.io/mkdocs-material/reference/)

---

## 🔗 Related Links

- [MkDocs Material Docs](https://squidfunk.github.io/mkdocs-material/)
- [MkDocs Docs](https://www.mkdocs.org/)
- [Cloudflare Pages Docs](https://developers.cloudflare.com/pages/)

---

> 🌐 Language: **English** · [切换到中文版](./assets/README.zh-CN.md)
