from __future__ import annotations

from typing import Any, Protocol

from localmind.config import DeviceMode


class ChatModel(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        enable_thinking: bool,
    ) -> str:
        ...


class TransformersChatModel:
    def __init__(
        self,
        model_name: str,
        device: DeviceMode = "auto",
        console: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.console = console or PlainConsole()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device_summary = "not loaded"

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        enable_thinking: bool,
    ) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        inputs = self._tokenizer.apply_chat_template(
            messages,
            xml_tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )
        output_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing model dependencies. Run `uv sync` before starting LocalMind."
            ) from exc

        cuda_available = torch.cuda.is_available()
        device_map: str | dict[str, str]

        if self.device == "cuda":
            if not cuda_available:
                torch_cuda = getattr(torch.version, "cuda", None)
                raise RuntimeError(
                    "CUDA was requested with --device cuda, but PyTorch cannot access CUDA. "
                    f"Installed torch: {torch.__version__}; torch CUDA build: {torch_cuda}. "
                    "Install a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ "
                    "or use --device cpu."
                )
            device_map = "auto"
            gpu_name = torch.cuda.get_device_name(0)
            self._device_summary = f"cuda ({gpu_name})"
        elif self.device == "cpu":
            device_map = {"": "cpu"}
            self._device_summary = "cpu"
        else:
            if cuda_available:
                device_map = "auto"
                gpu_name = torch.cuda.get_device_name(0)
                self._device_summary = f"auto -> cuda ({gpu_name})"
            else:
                device_map = {"": "cpu"}
                self._device_summary = "auto -> cpu"

        if self.device == "auto" and not cuda_available:
            self.console.print(
                "[yellow]CUDA was not detected; using CPU. SmolLM3-3B may be slow.[/yellow]"
            )

        self.console.print("[bold cyan]Model setup[/bold cyan]")
        self.console.print(f"[dim]Requested device:[/dim] {self.device}")
        self.console.print(f"[dim]Resolved device:[/dim] {self._device_summary}")
        self.console.print(f"[dim]Model:[/dim] {self.model_name}")
        self.console.print("[dim]Loading model weights...[/dim]")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype="auto",
        )
        self.console.print("[green]Model ready.[/green]")

    @property
    def device_summary(self) -> str:
        return self._device_summary


class PlainConsole:
    def print(self, message: str) -> None:
        print(message)
