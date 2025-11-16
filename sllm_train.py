from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from datasets import load_dataset
from transformers import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sllm_model import LLM
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TrainConfig:
    epochs: int = 20
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    ff_hidden_dim: Optional[int] = 3072
    base_lr: float = 5e-4
    lr_scale_reference_batch: int = 4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    min_lr: float = 1e-5
    batch_size: int = 4
    block_size: int = 1024
    num_workers: int = 4
    prefetch_factor: Optional[int] = 2
    max_new_tokens: int = 50
    prompt: str = "Once upon a time"
    step_size: float = 1.0
    initial_velocity: str = "linear"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    tokenizer_name: str = "gpt2"
    seed: int = 42
    grad_clip_norm: float = 1.0
    precision: str = "bf16-true"
    log_every_n_steps: int = 1
    use_autocast: bool = True
    autocast_device_type: Optional[str] = None
    autocast_dtype: Optional[str] = None


def _tokenize_batch(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    return tokenizer(examples["text"])


def _filter_block_size(example: Dict[str, List[int]], block_size: int) -> bool:
    return len(example["input_ids"]) == block_size


def _group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, Any]:
    concatenated = sum(examples["input_ids"], [])
    concatenated_attention = sum(examples["attention_mask"], [])
    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": [], "attention_mask": []}

    chunks = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
    attn_chunks = [concatenated_attention[i : i + block_size] for i in range(0, total_length, block_size)]
    return {"input_ids": chunks, "attention_mask": attn_chunks}


class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset = None
        try:
            self.tokenizer.disable_parallelism()
        except AttributeError:
            pass

    def prepare_data(self) -> None:
        load_dataset(self.cfg.dataset_name, self.cfg.dataset_config)
        AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)

    def setup(self, stage: Optional[str] = None) -> None:
        raw_datasets = load_dataset(self.cfg.dataset_name, self.cfg.dataset_config)

        tokenized = raw_datasets.map(
            _tokenize_batch,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            fn_kwargs={"tokenizer": self.tokenizer},
            load_from_cache_file=True,
        )

        lm_datasets = tokenized.map(
            _group_texts,
            batched=True,
            fn_kwargs={"block_size": self.cfg.block_size},
            load_from_cache_file=True,
        )
        lm_datasets = lm_datasets.filter(
            _filter_block_size,
            fn_kwargs={"block_size": self.cfg.block_size},
            load_from_cache_file=True,
        )

        split = lm_datasets["train"].train_test_split(test_size=0.05, seed=42)
        val_dataset = lm_datasets["validation"]
        self.dataset = {
            "train": split["train"],
            "validation": val_dataset,
            "eval": split["test"],
        }

    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _build_loader(self, dataset, *, shuffle: bool) -> DataLoader:
        loader_kwargs = {
            "batch_size": self.cfg.batch_size,
            "shuffle": shuffle,
            "num_workers": self.cfg.num_workers,
            "collate_fn": self.collate_fn,
            "persistent_workers": self.cfg.num_workers > 0,
            "drop_last": True,
        }
        if self.cfg.num_workers > 0 and self.cfg.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.cfg.prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.dataset["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self.dataset["validation"], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(self.dataset["eval"], shuffle=False)


class AutoregressiveLLMModule(pl.LightningModule):
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ff_hidden_dim: Optional[int],
        lr: float,
        weight_decay: float,
        pad_token_id: int,
        step_size: float,
        initial_velocity: str,
        context_length: int,
        warmup_steps: int,
        min_lr: float,
        tokenizer: Optional[AutoTokenizer],
        sample_prompt: str,
        sample_max_new_tokens: int,
        precision_mode: str,
        use_autocast: bool,
        autocast_device_type: Optional[str],
        autocast_dtype: Optional[str],
    ):
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length must be a positive integer.")
        self.model = LLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            step_size=step_size,
            initial_velocity=initial_velocity,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.pad_token_id = pad_token_id
        self.context_length = context_length
        self.warmup_steps = max(0, warmup_steps)
        self.min_lr = max(0.0, min_lr)
        self.sample_tokenizer = tokenizer
        self.sample_prompt = sample_prompt
        self.sample_max_new_tokens = sample_max_new_tokens
        self.precision_mode = precision_mode.lower()
        self.autocast_enabled = use_autocast
        self.autocast_device_type = autocast_device_type
        self.autocast_dtype = self._resolve_autocast_dtype(autocast_dtype)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine_warmup",
            },
        }

    def _resolve_autocast_dtype(self, dtype_name: Optional[str]) -> Optional[torch.dtype]:
        if dtype_name is None:
            return None
        name = dtype_name.lower()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return mapping.get(name, None)

    def _autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        param = next(self.model.parameters(), None)
        if param is None:
            return nullcontext()
        param_device = param.device
        device_type = self.autocast_device_type or param_device.type
        if device_type == "xla":
            return nullcontext()
        dtype = self.autocast_dtype
        if dtype is None:
            if "bf16" in self.precision_mode:
                dtype = torch.bfloat16
            elif "16" in self.precision_mode:
                dtype = torch.float16
            else:
                dtype = torch.float32
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)

    def _warmup_rotary_cache(self) -> None:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.model.warmup_rotary_cache(self.context_length, device=device, dtype=dtype)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[LambdaLR]:
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None or total_steps <= 0:
            return None
        if total_steps == 1:
            return None
        warmup_steps = min(self.warmup_steps, total_steps - 1)
        decay_steps = max(1, total_steps - warmup_steps)
        base_lr = self.lr
        min_lr_factor = self.min_lr / base_lr if base_lr > 0 else 0.0
        min_lr_factor = min(max(min_lr_factor, 0.0), 1.0)

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (cosine * (1.0 - min_lr_factor)) + min_lr_factor

        return LambdaLR(optimizer, lr_lambda)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._warmup_rotary_cache()

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._warmup_rotary_cache()

    def on_test_start(self) -> None:
        super().on_test_start()
        self._warmup_rotary_cache()

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        padding = attention_mask[:, :-1]

        with self._autocast_context():
            logits = self.model(inputs, padding_mask=padding)
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=self.pad_token_id,
            )
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=(stage == "train"),
            batch_size=inputs.size(0),
        )
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "val")
        perplexity = torch.exp(loss.detach())
        self.log("val_perplexity", perplexity, prog_bar=True, on_epoch=True, batch_size=batch["input_ids"].size(0))
        return loss

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if self.sample_tokenizer is None:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        try:
            text = self.generate(
                self.sample_tokenizer,
                self.sample_prompt,
                max_new_tokens=self.sample_max_new_tokens,
            )
        except ValueError as exc:
            self.print(f"[generation skipped] {exc}")
            return
        self.print(f"=== Sampled Text (epoch {self.current_epoch}) ===\n{text}")

    def generate(self, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 50) -> str:
        was_training = self.training
        try:
            self.eval()
            device = next(self.parameters()).device
            with torch.no_grad():
                tokens = tokenizer(prompt, return_tensors="pt")
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)
                prompt_len = input_ids.size(1)
                total_len = self.context_length
                if prompt_len >= total_len:
                    raise ValueError(
                        f"Prompt is too long for the configured context window ({self.context_length}). "
                        "Increase block_size to generate from this prompt."
                    )
                available_steps = total_len - prompt_len
                steps_to_generate = min(max_new_tokens, available_steps)
                # Warm up the rotary cache for the current device/dtype combo before decoding.
                self.model.warmup_rotary_cache(self.context_length, device=device, dtype=next(self.parameters()).dtype)
                batch_size = input_ids.size(0)
                padded_input = torch.full(
                    (batch_size, total_len),
                    fill_value=self.pad_token_id,
                    dtype=input_ids.dtype,
                    device=device,
                )
                padded_mask = torch.zeros((batch_size, total_len), dtype=attention_mask.dtype, device=device)
                padded_input[:, :prompt_len] = input_ids
                padded_mask[:, :prompt_len] = attention_mask
                current_length = prompt_len
                for _ in range(steps_to_generate):
                    logits = self.model(padded_input, padding_mask=padded_mask)
                    next_token_logits = logits[:, current_length - 1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    padded_input[:, current_length] = next_token
                    padded_mask[:, current_length] = 1
                    current_length += 1
                text = tokenizer.batch_decode(padded_input[:, :current_length], skip_special_tokens=True)[0]
            return text
        finally:
            if was_training:
                self.train()


def main() -> None:
    train_cfg = TrainConfig()

    pl.seed_everything(train_cfg.seed, workers=True)
    data_module = WikiTextDataModule(train_cfg)
    data_module.setup()

    vocab_size = data_module.tokenizer.vocab_size
    lr_scale = train_cfg.batch_size / max(1, train_cfg.lr_scale_reference_batch)
    effective_lr = train_cfg.base_lr * lr_scale
    effective_min_lr = train_cfg.min_lr * lr_scale
    if effective_min_lr > effective_lr:
        effective_min_lr = effective_lr

    module = AutoregressiveLLMModule(
        vocab_size=vocab_size,
        embed_dim=train_cfg.embed_dim,
        depth=train_cfg.depth,
        num_heads=train_cfg.num_heads,
        ff_hidden_dim=train_cfg.ff_hidden_dim,
        lr=effective_lr,
        weight_decay=train_cfg.weight_decay,
        pad_token_id=data_module.pad_token_id,
        step_size=train_cfg.step_size,
        initial_velocity=train_cfg.initial_velocity,
        context_length=train_cfg.block_size,
        warmup_steps=train_cfg.warmup_steps,
        min_lr=effective_min_lr,
        tokenizer=data_module.tokenizer,
        sample_prompt=train_cfg.prompt,
        sample_max_new_tokens=train_cfg.max_new_tokens,
        precision_mode=train_cfg.precision,
        use_autocast=train_cfg.use_autocast,
        autocast_device_type=train_cfg.autocast_device_type,
        autocast_dtype=train_cfg.autocast_dtype,
    )

    resume_checkpoint_cb = ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
        save_weights_only=False,
        filename="resume",
        auto_insert_metric_name=False,
    )
    best_model_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        save_weights_only=True,
        filename="best",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=train_cfg.epochs,
        log_every_n_steps=train_cfg.log_every_n_steps,
        callbacks=[resume_checkpoint_cb, best_model_cb],
        precision=train_cfg.precision,
        gradient_clip_val=train_cfg.grad_clip_norm,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(module, datamodule=data_module)
    val_metrics = trainer.validate(module, datamodule=data_module)

    if val_metrics:
        loss = val_metrics[0].get("val_loss")
        if loss is not None and math.isfinite(loss):
            perplexity = math.exp(loss)
            print(f"Validation Loss: {loss:.4f} | Perplexity: {perplexity:.4f}")
        else:
            print("Validation metrics unavailable or non-finite.")

    generated = module.generate(
        data_module.tokenizer,
        train_cfg.prompt,
        max_new_tokens=train_cfg.max_new_tokens,
    )
    print("=== Sampled Text ===")
    print(generated)


if __name__ == "__main__":
    main()
