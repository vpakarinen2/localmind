# LocalMind

A small local CLI agent built around [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

## Features

- Local chat with SmolLM3-3B through Hugging Face Transformers
- Workspace that keeps local activity within selected directory
- Built-in tools (calculator, local date and time, file read/write)
- Device selection with `auto`, `cpu`, and `cuda` modes
- Optional web search through self-hosted SearXNG
- Session-only memory; no chat history is persisted

## Requirements

- NVIDIA GPU with CUDA-enabled PyTorch (Optional)
- Docker if you want to run SearXNG locally (Optional)
- Python 3.12 is recommended
- [`uv`](https://docs.astral.sh/uv/)

## Example

```
you: Who was the US senator who died recently? Use a web search.
Model setup
Requested device: auto
Resolved device: auto -> cuda (NVIDIA GeForce RTX 4060 Laptop GPU)
Model: HuggingFaceTB/SmolLM3-3B
Loading model weights...
Model ready.

LocalMind
U.S. Senator Lindsey Graham, a close ally of President Donald Trump, has died at the age of 71 after a brief and sudden illness, according to his office. He was a Republican from South Carolina.
```

## Installation

```powershell
uv sync
```

## Usage

Start chat session:

```powershell
uv run localmind chat --workspace ./workspace
```

Use CUDA when available:

```powershell
uv run localmind chat --workspace ./workspace --device cuda
```

### Workspace

Workspace is a local working directory that defaults to `./workspace`. 

You can optionally change it with `--workspace`:

```
uv run localmind chat --workspace .\workspace\my-session
```

### Keywords

#### Response Keywords

```
Numbered list: numbered list
Bullet list: bullet list
Recent News: latest, news, or updates
Paragraph: one paragraph or single paragraph
Multiple paragraphs: 3 paragraphs
Top Result: top 5, top five
Markdown: Markdown
```

## Web Search

LocalMind uses web search through SearXNG:

```powershell
uv run localmind chat --workspace ./workspace --search --searxng-url http://localhost:8080
```

### Run SearXNG Locally

From the project root, start SearXNG:

```cmd
docker run --rm -d --name localmind-searxng -p 8080:8080 --mount type=bind,source="%cd%\searxng",target=/etc/searxng -e BASE_URL=http://localhost:8080/ searxng/searxng:latest
```

### SearXNG Settings

```
use_default_settings: true

server:
  secret_key: "<secret_key_here>"
  image_proxy: true
  bind_address: "0.0.0.0"
  port: 8080

search:
  formats:
    - html
    - json
```

## CUDA Setup

Check the installed PyTorch:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Install CUDA-enabled PyTorch:

```powershell
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## License

LocalMind is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).

## Author

Ville Pakarinen (@vpakarinen2)
