# LocalMind

Terminal-first small local Python agent built around [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

## Features

- Local chat with SmolLM3-3B through Hugging Face Transformers
- Built-in tools (calculator, local date and time, file read/write)
- Device selection with `auto`, `cpu`, and `cuda` modes
- Optional web search through self-hosted SearXNG
- Session-only memory; no chat history is persisted

## Requirements

- NVIDIA GPU with CUDA-enabled PyTorch (Optional)
- Docker if you want to run SearXNG locally (Optional)
- Python 3.12 is recommended
- [`uv`](https://docs.astral.sh/uv/)

CPU mode works, but SmolLM3-3B is much faster with CUDA.

## Installation

```powershell
uv sync
```

## Usage

Start a chat session:

```powershell
uv run localmind chat --workspace ./workspace
```

Use CUDA when available:

```powershell
uv run localmind chat --workspace ./workspace --device auto
```

Useful chat commands:

- `/help`
- `/exit`

## Web Search

LocalMind can use web search through SearXNG:

```powershell
uv run localmind chat --workspace ./workspace --search --searxng-url http://localhost:8080
```

### Run SearXNG Locally

Minimal config file at
`searxng/settings.yml` with JSON output enabled.

From the project root, start SearXNG:

```powershell
docker stop localmind-searxng
docker rm localmind-searxng

docker run --rm -d `
  --name localmind-searxng `
  -p 8080:8080 `
  --mount type=bind,source="${PWD}\searxng",target=/etc/searxng `
  -e BASE_URL=http://localhost:8080/ `
  searxng/searxng:latest
```

If you use `cmd`, use `%cd%` instead of `${PWD}`:

```cmd
docker run --rm -d --name localmind-searxng -p 8080:8080 --mount type=bind,source="%cd%\searxng",target=/etc/searxng -e BASE_URL=http://localhost:8080/ searxng/searxng:latest
```

## CUDA Setup

Check the installed PyTorch build:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Install CUDA-enabled PyTorch wheel:

```powershell
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## License

LocalMind is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).

## Author

Ville Pakarinen (@vpakarinen2)
