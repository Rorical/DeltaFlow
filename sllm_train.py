from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from sllm_model import LLM
except ModuleNotFoundError:
    pass
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Kaggle TPUs export TPU_PROCESS_ADDRESSES=local, but torch_xla expects either
# an empty value or one entry per TPU slice. Drop the placeholder to unblock PJRT.
if os.environ.get("TPU_PROCESS_ADDRESSES") == "local":
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)


@dataclass
class TrainConfig:
    epochs: int = 5
    embed_dim: int = 400
    depth: int = 12
    num_heads: int = 8
    ff_hidden_dim: Optional[int] = 1760
    base_lr: float = 2e-4
    weight_decay: float = 0.02
    batch_size: int = 16
    grad_accum_steps: int = 1
    block_size: int = 512
    num_workers: Optional[int] = None
    prefetch_factor: Optional[int] = 2
    max_new_tokens: int = 100
    prompt: str = "On its day of"
    step_size: float = 1.0
    initial_velocity: str = "linear"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    tokenizer_name: str = "gpt2"
    train_subset_size: Optional[int] = None
    val_subset_size: Optional[int] = 4096
    test_subset_size: Optional[int] = 4096
    seed: int = 42
    grad_clip_norm: float = 5.0
    precision: str = "32-true"
    log_every_n_steps: int = 1
    use_autocast: bool = True
    autocast_device_type: Optional[str] = None
    autocast_dtype: Optional[str] = None
    auto_resume: bool = False
    resume_checkpoint_name: str = "resume.ckpt"


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

        train_dataset = lm_datasets["train"]
        val_dataset = lm_datasets.get("validation")
        test_dataset = lm_datasets.get("test")

        if val_dataset is None:
            raise RuntimeError("Validation split missing from dataset; please provide one for evaluation.")
        if test_dataset is None:
            split = train_dataset.train_test_split(test_size=0.05, seed=self.cfg.seed)
            train_dataset = split["train"]
            test_dataset = split["test"]

        self.dataset = {
            "train": self._maybe_truncate(train_dataset, self.cfg.train_subset_size),
            "validation": self._maybe_truncate(val_dataset, self.cfg.val_subset_size),
            "test": self._maybe_truncate(test_dataset, self.cfg.test_subset_size),
        }

    def _maybe_truncate(self, dataset, max_samples: Optional[int]):
        if max_samples is None:
            return dataset
        max_samples = max(0, max_samples)
        if max_samples == 0 or len(dataset) <= max_samples:
            if max_samples == 0:
                return dataset.select([])
            return dataset
        return dataset.select(list(range(max_samples)))

    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _build_loader(self, dataset, *, shuffle: bool) -> DataLoader:
        num_workers = self.cfg.num_workers
        if num_workers is None:
            try:
                num_workers = min(8, os.cpu_count() or 1)
            except NotImplementedError:
                num_workers = 0
        loader_kwargs = {
            "batch_size": self.cfg.batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": self.collate_fn,
            "persistent_workers": num_workers > 0,
            "drop_last": True,
        }
        if num_workers > 0 and self.cfg.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.cfg.prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.dataset["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self.dataset["validation"], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(self.dataset["test"], shuffle=False)


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
        self.sample_tokenizer = tokenizer
        self.sample_prompt = sample_prompt
        self.sample_max_new_tokens = sample_max_new_tokens
        self.precision_mode = precision_mode.lower()
        self.autocast_enabled = use_autocast
        self.autocast_device_type = autocast_device_type
        self.autocast_dtype = self._resolve_autocast_dtype(autocast_dtype)
        self._val_example_batch: Optional[Dict[str, torch.Tensor]] = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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

        if stage == "val" and self._val_example_batch is None:
            trainer = getattr(self, "trainer", None)
            is_global_zero = True if trainer is None else trainer.is_global_zero
            if is_global_zero:
                self._val_example_batch = {k: v.detach().clone() for k, v in batch.items()}

        with self._autocast_context():
            logits = self.model(inputs, padding_mask=padding)
            vocab_size = logits.size(-1)
            targets_flat = targets.reshape(-1)
            padding_flat = padding.reshape(-1)
            ignore_mask = padding_flat == 0
            if ignore_mask.any():
                targets_flat = targets_flat.masked_fill(ignore_mask, -100)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets_flat,
                ignore_index=-100,
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
        if self.trainer is not None and not self.trainer.is_global_zero:
            self._val_example_batch = None
            return
        if self.sample_tokenizer is not None:
            try:
                text = self.generate(
                    self.sample_tokenizer,
                    self.sample_prompt,
                    max_new_tokens=self.sample_max_new_tokens,
                )
            except ValueError as exc:
                self.print(f"[generation skipped] {exc}")
            else:
                self.print(f"=== Sampled Text (epoch {self.current_epoch}) ===\n{text}")
        self._log_second_order_diagnostics()
        self._val_example_batch = None

    def _stack_param_norm(self, tensors: List[torch.Tensor]) -> float:
        if not tensors:
            return 0.0
        norms = []
        for tensor in tensors:
            norms.append(tensor.detach().float().norm().cpu())
        return float(torch.stack(norms).mean())

    def _log_second_order_diagnostics(self) -> None:
        """
        After each epoch, check whether the second-order path is active.
        Prints:
          - Per-layer step sizes (should vary; all near 1e-3 means near-first-order).
          - Per-layer norms for x, v, accel_attn, accel_ff on a held-out val batch.
          - Mean parameter norms for velocity-specific weights.
        TPU/XLA: small host syncs via .cpu() are acceptable here since this runs once per epoch.
        """
        if self._val_example_batch is None:
            self.print("[second-order probe skipped] no cached validation batch.")
            return

        device = next(self.parameters()).device
        padding_mask = self._val_example_batch["attention_mask"].to(device)
        tokens = self._val_example_batch["input_ids"].to(device)
        seq_len = tokens.size(1)
        mask = self.model._build_attention_mask(padding_mask, seq_len, device)

        with torch.no_grad():
            embeddings = self.model.token_embedding(tokens)
            _, _, layer_stats = self.model.backbone.forward_with_stats(embeddings, mask=mask)

        steps = [stat["step"] for stat in layer_stats]
        x_norms = [stat["x_norm"] for stat in layer_stats]
        v_norms = [stat["v_norm"] for stat in layer_stats]
        accel_attn_norms = [stat["accel_attn_norm"] for stat in layer_stats]
        accel_ff_norms = [stat["accel_ff_norm"] for stat in layer_stats]
        damping_vals = [stat.get("damping") for stat in layer_stats if "damping" in stat]

        attn_layers = [layer.attention for layer in self.model.backbone.layers]
        ff_layers = [layer.feed_forward for layer in self.model.backbone.layers]
        fused_param_norms = {
            "attn_q_proj": self._stack_param_norm([layer.q_proj.weight for layer in attn_layers]),
            "attn_k_proj": self._stack_param_norm([layer.k_proj.weight for layer in attn_layers]),
            "attn_v_proj": self._stack_param_norm([layer.v_proj.weight for layer in attn_layers]),
            "ff_w1": self._stack_param_norm([layer.w1.weight for layer in ff_layers]),
            "ff_w2": self._stack_param_norm([layer.w2.weight for layer in ff_layers]),
        }

        step_min, step_max = min(steps), max(steps)
        v_over_x = float(torch.tensor(v_norms).mean() / max(torch.tensor(x_norms).mean(), 1e-6))
        lines = [
            "[second-order probe] step sizes:"
            f" min={step_min:.4e}, max={step_max:.4e}, mean={float(torch.tensor(steps).mean()):.4e}",
            f"[second-order probe] steps per layer: {', '.join(f'{s:.4e}' for s in steps)}",
            "[second-order probe] norms (mean over batch/seq per layer): "
            f"x={float(torch.tensor(x_norms).mean()):.4f}, "
            f"v={float(torch.tensor(v_norms).mean()):.4f} (v/x={v_over_x:.4f}), "
            f"accel_attn={float(torch.tensor(accel_attn_norms).mean()):.4f}, "
            f"accel_ff={float(torch.tensor(accel_ff_norms).mean()):.4f}",
            "[second-order probe] x_norm per layer: " + ", ".join(f"{n:.4f}" for n in x_norms),
            "[second-order probe] v_norm per layer: " + ", ".join(f"{n:.4f}" for n in v_norms),
            "[second-order probe] damping per layer: " + ", ".join(f"{d:.4f}" for d in damping_vals) if damping_vals else "[second-order probe] damping per layer: n/a",
            "[second-order probe] fused param norms (mean over layers): "
            + ", ".join(f"{k}={v:.4f}" for k, v in fused_param_norms.items()),
        ]
        for line in lines:
            self.print(line)

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

    module = AutoregressiveLLMModule(
        vocab_size=vocab_size,
        embed_dim=train_cfg.embed_dim,
        depth=train_cfg.depth,
        num_heads=train_cfg.num_heads,
        ff_hidden_dim=train_cfg.ff_hidden_dim,
        lr=train_cfg.base_lr,
        weight_decay=train_cfg.weight_decay,
        pad_token_id=data_module.pad_token_id,
        step_size=train_cfg.step_size,
        initial_velocity=train_cfg.initial_velocity,
        context_length=train_cfg.block_size,
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
        accumulate_grad_batches=max(1, train_cfg.grad_accum_steps),
    )
    ckpt_path = None
    if train_cfg.auto_resume:
        ckpt_candidate = os.path.join(os.getcwd(), train_cfg.resume_checkpoint_name)
        if os.path.exists(ckpt_candidate):
            ckpt_path = ckpt_candidate
            print(f"[auto-resume] Loading checkpoint from {ckpt_path}")
        else:
            print(f"[auto-resume] No checkpoint found at {ckpt_candidate}, starting fresh.")

    trainer.fit(module, datamodule=data_module, ckpt_path=ckpt_path)
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
