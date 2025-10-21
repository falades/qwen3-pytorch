## About

A minimal PyTorch re-implementation of `Qwen3`

## 🦋 Quick Start

I recommend using `uv` and creating a virtual environment:

```bash
pip install uv && uv venv

# activate the environment
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows

# install dependencies
uv pip install -r requirements.txt
```


Usage:

```bash
python run.py --prompt "Give me a short introduction to large language models."
```

The default model is Qwen3-0.6B-Base to pass other qwen3 based models just pass the huggingface repo id as a flag

```bash
python run.py --prompt "Give me a short introduction to large language models." --repo_id "Qwen/Qwen3-1.7B-Base"
```