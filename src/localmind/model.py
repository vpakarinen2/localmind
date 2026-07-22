from __future__ import annotations

from typing import Any, Protocol

from localmind.config import DeviceMode, PromptFormat


ALPACA_SYSTEM_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request."
)
ALPACA_SYSTEM_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request."
)
SEARCH_DATA_MARKER = "BEGIN_UNTRUSTED_WEB_SEARCH_DATA"
SEARCH_TRANSCRIPT_STOP_STRINGS = (
    "### Input:",
    "### Instruction:",
    SEARCH_DATA_MARKER,
)


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
        lora_model: str | None = None,
        prompt_format: PromptFormat = "chat",
        device: DeviceMode = "auto",
        console: Any | None = None,
        max_new_tokens: int = 1024,
        thinking_max_new_tokens: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.lora_model = lora_model
        self.prompt_format = prompt_format
        self.device = device
        self.console = console or PlainConsole()
        self.max_new_tokens = max(1, max_new_tokens)
        self.thinking_max_new_tokens = max(1, thinking_max_new_tokens)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device_summary = "not loaded"
        self._adapter_summary = "none"

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        enable_thinking: bool,
    ) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        self._tokenizer.truncation_side = "left"
        generation_limit = (
            self.thinking_max_new_tokens if enable_thinking else self.max_new_tokens
        )
        context_window = getattr(
            getattr(self._model, "config", None), "max_position_embeddings", 32_768
        )
        if not isinstance(context_window, int) or context_window < 2:
            context_window = 32_768
        if generation_limit >= context_window:
            raise ValueError(
                f"The selected generation limit ({generation_limit}) must be smaller than "
                f"the model context window ({context_window})."
            )
        max_input_tokens = max(1, context_window - generation_limit)
        if self.prompt_format == "alpaca":
            prompt = self._format_alpaca_prompt(messages)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            ).to(self._model.device)
        else:
            inputs = self._tokenizer.apply_chat_template(
                messages,
                xml_tools=tools,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            ).to(self._model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": generation_limit,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
        }
        has_search_context = any(
            message.get("role") == "tool"
            and SEARCH_DATA_MARKER in message.get("content", "")
            for message in messages
        )
        if has_search_context:
            generation_kwargs.update(
                stop_strings=list(SEARCH_TRANSCRIPT_STOP_STRINGS),
                tokenizer=self._tokenizer,
            )
        outputs = self._model.generate(**inputs, **generation_kwargs)
        output_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        if len(output_ids) >= generation_limit:
            self.console.print(
                "[yellow]Generation reached its token limit and may be incomplete. "
                "Increase the applicable max-new-tokens option if needed.[/yellow]"
            )
        return self._tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    @staticmethod
    def _format_alpaca_prompt(messages: list[dict[str, str]]) -> str:
        conversation = [message for message in messages if message.get("role") != "system"]
        latest_user_index = next(
            (
                index
                for index in range(len(conversation) - 1, -1, -1)
                if conversation[index].get("role") == "user"
            ),
            None,
        )
        if latest_user_index is None:
            raise ValueError("Alpaca prompt formatting requires at least one user message.")

        instruction = conversation[latest_user_index].get("content", "").strip()
        context_messages = [
            message for index, message in enumerate(conversation) if index != latest_user_index
        ]
        if not context_messages:
            return (
                f"{ALPACA_SYSTEM_NO_INPUT}\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )

        role_names = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
        context = "\n\n".join(
            f"{role_names.get(message.get('role', ''), message.get('role', 'Context').title())}: "
            f"{message.get('content', '').strip()}"
            for message in context_messages
        )
        return (
            f"{ALPACA_SYSTEM_WITH_INPUT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n### Response:\n"
        )

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
        if self.lora_model:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA adapter support requires PEFT. Run `uv sync` to install it."
                ) from exc

            self.console.print(f"[dim]LoRA adapter:[/dim] {self.lora_model}")
            self.console.print("[dim]Loading LoRA adapter weights...[/dim]")
            try:
                self._model = PeftModel.from_pretrained(self._model, self.lora_model)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load LoRA adapter '{self.lora_model}' on base model "
                    f"'{self.model_name}': {exc}"
                ) from exc
            self._adapter_summary = self._get_active_adapter_summary(self._model)
            self.console.print(
                f"[green]Active LoRA adapter(s):[/green] {self._adapter_summary}"
            )
        self.console.print("[green]Model ready.[/green]")

    @staticmethod
    def _get_active_adapter_summary(model: Any) -> str:
        try:
            active = getattr(model, "active_adapters", None)
            if callable(active):
                active = active()
            if active is None:
                active = getattr(model, "active_adapter", None)
            if isinstance(active, str):
                return active
            if active:
                return ", ".join(str(name) for name in active)
        except Exception:
            pass
        return "loaded (PEFT did not expose adapter names)"

    @property
    def device_summary(self) -> str:
        return self._device_summary

    @property
    def adapter_summary(self) -> str:
        return self._adapter_summary


class PlainConsole:
    def print(self, message: str) -> None:
        print(message)
