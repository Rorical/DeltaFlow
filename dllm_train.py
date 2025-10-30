from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

import pytorch_lightning as pl

from model import DLLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    tokenizer_name: str = "gpt2"
    block_size: int = 128
    batch_size: int = 2
    num_workers: int = 0


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
    def __init__(self, data_cfg: DataConfig):
        super().__init__()
        self.cfg = data_cfg
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
            load_from_cache_file=False,
        )

        lm_datasets = tokenized.map(
            _group_texts,
            batched=True,
            fn_kwargs={"block_size": self.cfg.block_size},
        )
        lm_datasets = lm_datasets.filter(
            _filter_block_size,
            fn_kwargs={"block_size": self.cfg.block_size},
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["eval"],
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.cfg.num_workers > 0,
        )


class DiffusionLLMModule(pl.LightningModule):
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ff_hidden_dim: Optional[int],
        lr: float,
        pad_token_id: int,
        sample_interval: int = 1000,
        sample_seq_len: int = 64,
        sample_prompt: Optional[str] = None,
    ):
        super().__init__()
        self.model = DLLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            pad_token_id=pad_token_id,
        )
        self.lr = lr
        self.pad_token_id = pad_token_id
        self.sample_interval = sample_interval
        self.sample_seq_len = sample_seq_len
        self.sample_prompt = sample_prompt
        self.tokenizer: Optional[AutoTokenizer] = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _sample_timesteps(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, seq_len, device=device)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        token_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        t = self._sample_timesteps(*token_ids.shape, token_ids.device)
        loss, components = self.model.loss(
            token_ids,
            timesteps=t,
            padding_mask=attention_mask,
            return_components=True,
        )
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=(stage == "train"),
            batch_size=token_ids.size(0),
        )
        for name, value in components.items():
            if stage == "train":
                on_step = name in {"loss_flow", "loss_recon_term"}
                prog_bar = name in {"loss_flow", "loss_recon_term"}
            else:
                on_step = False
                prog_bar = False

            self.log(
                f"{stage}_{name}",
                value,
                prog_bar=prog_bar,
                on_epoch=True,
                on_step=on_step,
                batch_size=token_ids.size(0),
            )
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        global_step = self.global_step
        if (
            self.tokenizer is not None
            and self.sample_interval > 0
            and global_step % self.sample_interval == 0
        ):
            with torch.no_grad():
                conditional_tokens = None
                if self.sample_prompt and self.tokenizer is not None:
                    encoded = self.tokenizer(
                        self.sample_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.sample_seq_len,
                    )
                    cond_ids = encoded["input_ids"].to(loss.device)
                    if cond_ids.size(1) < self.sample_seq_len:
                        cond_full = torch.full(
                            (cond_ids.size(0), self.sample_seq_len),
                            self.tokenizer.pad_token_id,
                            device=loss.device,
                            dtype=torch.long,
                        )
                        cond_full[:, : cond_ids.size(1)] = cond_ids
                    else:
                        cond_full = cond_ids[:, : self.sample_seq_len]
                    conditional_tokens = cond_full

                tokens = self.model.sample(
                    batch_size=1,
                    seq_len=self.sample_seq_len,
                    steps=20,
                    device=loss.device,
                    sampler="heun",
                    conditional_tokens=conditional_tokens,
                )
                text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
                self.log("train_sample_len", torch.tensor(len(text), device=loss.device))
                self.print(f"\n[Sample @ step {global_step}] {text}\n")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "val")
        with torch.no_grad():
            token_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            t = torch.ones_like(token_ids, dtype=torch.float32, device=loss.device) * 0.9

            target_logits = self.model._build_target_logits(token_ids).detach()
            init_logits = torch.randn_like(target_logits) * self.model.init_scale
            state_logits = (1.0 - t.unsqueeze(-1)) * init_logits + t.unsqueeze(-1) * target_logits
            delta = (1.0 - t).unsqueeze(-1)

            velocity = self.model.predict_velocity(state_logits, timesteps=t.unsqueeze(-1), padding_mask=attention_mask)
            pred_logits = state_logits + delta * velocity
            preds = pred_logits.argmax(dim=-1)
            mask = attention_mask.bool()
            correct = (preds == token_ids).masked_fill(~mask, False).sum()
            total = mask.sum()
            acc = correct.float() / total.clamp_min(1)
            self.log("val_accuracy", acc, prog_bar=True, on_epoch=True, batch_size=token_ids.size(0))
        return loss

    def sample(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int,
        *,
        steps: int = 10,
        prompt: Optional[str] = None,
        sampler: str = "heun",
    ) -> str:
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            conditional_tokens = None
            if prompt:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=seq_len,
                )
                cond_ids = encoded["input_ids"].to(device)
                prompt_len = cond_ids.size(1)
                if prompt_len < seq_len:
                    cond_full = torch.full(
                        (cond_ids.size(0), seq_len),
                        tokenizer.pad_token_id,
                        device=device,
                        dtype=torch.long,
                    )
                    cond_full[:, :prompt_len] = cond_ids
                else:
                    cond_full = cond_ids[:, :seq_len]
                conditional_tokens = cond_full

            tokens = self.model.sample(
                batch_size=1,
                seq_len=seq_len,
                steps=steps,
                device=device,
                conditional_tokens=conditional_tokens,
                sampler=sampler,
            )
            text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DLLM on WikiText using PyTorch Lightning")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-hidden-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps during evaluation.")
    parser.add_argument("--prompt", type=str, default="Once upon", help="Prompt text for conditional sampling.")
    parser.add_argument("--sample-interval", type=int, default=0, help="Training sample interval in steps (0 to disable).")
    parser.add_argument("--sample-seq-len", type=int, default=64, help="Sequence length for periodic training samples.")
    parser.add_argument("--sample-prompt", type=str, default=None, help="Prompt used for periodic training samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pl.seed_everything(42, workers=True)
    data_cfg = DataConfig(batch_size=args.batch_size, block_size=args.block_size, num_workers=args.num_workers)
    data_module = WikiTextDataModule(data_cfg)
    data_module.setup()

    vocab_size = data_module.tokenizer.vocab_size

    module = DiffusionLLMModule(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        lr=args.lr,
        pad_token_id=data_module.pad_token_id,
        sample_interval=args.sample_interval,
        sample_seq_len=args.sample_seq_len,
        sample_prompt=args.sample_prompt or args.prompt,
    )
    module.tokenizer = data_module.tokenizer

    trainer = pl.Trainer(max_epochs=args.epochs, log_every_n_steps=1, enable_checkpointing=False)
    trainer.fit(module, datamodule=data_module)
    val_metrics = trainer.validate(module, datamodule=data_module)

    if val_metrics:
        metrics = val_metrics[0]
        loss = metrics.get("val_loss")
        acc = metrics.get("val_accuracy")
        msg_parts = []
        if loss is not None and math.isfinite(loss):
            msg_parts.append(f"Validation Loss: {loss:.4f}")
        if acc is not None and math.isfinite(acc):
            msg_parts.append(f"Accuracy: {acc:.4f}")
        if msg_parts:
            print(" | ".join(msg_parts))
        else:
            print("Validation metrics unavailable or non-finite.")

    generated = module.sample(
        data_module.tokenizer,
        seq_len=args.block_size,
        steps=args.steps,
        prompt=args.prompt,
        sampler="heun",
    )
    print("=== Sampled Text ===")
    print(generated)


if __name__ == "__main__":
    main()
