from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from model import DLLM
from transformers import AutoTokenizer

SAMPLE_TEXT = """
Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
While I nodded, nearly napping, suddenly there came a tapping,
As of someone gently rapping, rapping at my chamber door.
“’Tis some visitor,” I muttered, “tapping at my chamber door—
Only this and nothing more.”
Ah, distinctly I remember it was in the bleak December;
And each separate dying ember wrought its ghost upon the floor.
Eagerly I wished the morrow;—vainly I had sought to borrow
From my books surcease of sorrow—sorrow for the lost Lenore—
For the rare and radiant maiden whom the angels name Lenore—
Nameless here for evermore.
And the silken sad uncertain rustling of each purple curtain
Thrilled me—filled me with fantastic terrors never felt before;
So that now, to still the beating of my heart, I stood repeating
“’Tis some visitor entreating entrance at my chamber door—
Some late visitor entreating entrance at my chamber door;—
This it is and nothing more.”
Presently my soul grew stronger; hesitating then no longer,
“Sir,” said I, “or Madam, truly your forgiveness I implore;
But the fact is I was napping, and so gently you came rapping,
And so faintly you came tapping, tapping at my chamber door,
That I scarce was sure I heard you”—here I opened wide the door;—
Darkness there and nothing more.
Deep into that darkness peering, long I stood there wondering, fearing,
Doubting, dreaming dreams no mortal ever dared to dream before;
But the silence was unbroken, and the stillness gave no token,
And the only word there spoken was the whispered word, “Lenore?”
This I whispered, and an echo murmured back the word, “Lenore!”—
Merely this and nothing more.
Back into the chamber turning, all my soul within me burning,
Soon again I heard a tapping somewhat louder than before.
“Surely,” said I, “surely that is something at my window lattice;
Let me see, then, what thereat is, and explore this mystery more—
Let my heart be still a moment and this mystery explore;—
’Tis the wind and nothing more!”
Open here I flung the shutter, when, with many a flirt and flutter,
In there stepped a stately Raven of the saintly days of yore;
Not the least obeisance made he; not a minute stopped or stayed he;
But, with mien of lord or lady, perched above my chamber door—
Perched upon a bust of Pallas just above my chamber door—
Perched, and sat, and nothing more.
Then this ebony bird beguiling my sad fancy into smiling,
By the grave and stern decorum of the countenance it wore,
“Though thy crest be shorn and shaven, thou,” I said, “art sure no craven,
Ghastly grim and ancient Raven wandering from the Nightly shore—
Tell me what thy lordly name is on the Night’s Plutonian shore!”
Quoth the Raven “Nevermore.”
Much I marvelled this ungainly fowl to hear discourse so plainly,
Though its answer little meaning—little relevancy bore;
For we cannot help agreeing that no living human being
Ever yet was blessed with seeing bird above his chamber door—
Bird or beast upon the sculptured bust above his chamber door,
With such name as “Nevermore.”
But the Raven, sitting lonely on the placid bust, spoke only
That one word, as if his soul in that one word he did outpour.
Nothing further then he uttered—not a feather then he fluttered—
Till I scarcely more than muttered “Other friends have flown before—
On the morrow he will leave me, as my Hopes have flown before.”
Then the bird said “Nevermore.”
Startled at the stillness broken by reply so aptly spoken,
“Doubtless,” said I, “what it utters is its only stock and store
Caught from some unhappy master whom unmerciful Disaster
Followed fast and followed faster till his songs one burden bore—
Till the dirges of his Hope that melancholy burden bore
Of ‘Never—nevermore.’”
But the Raven still beguiling all my fancy into smiling,
Straight I wheeled a cushioned seat in front of bird, and bust and door;
Then, upon the velvet sinking, I betook myself to linking
Fancy unto fancy, thinking what this ominous bird of yore—
What this grim, ungainly, ghastly, gaunt, and ominous bird of yore
Meant in croaking “Nevermore.”
This I sat engaged in guessing, but no syllable expressing
To the fowl whose fiery eyes now burned into my bosom’s core;
This and more I sat divining, with my head at ease reclining
On the cushion’s velvet lining that the lamp-light gloated o’er,
But whose velvet-violet lining with the lamp-light gloating o’er,
She shall press, ah, nevermore!
Then, methought, the air grew denser, perfumed from an unseen censer
Swung by Seraphim whose foot-falls tinkled on the tufted floor.
“Wretch,” I cried, “thy God hath lent thee—by these angels he hath sent thee
Respite—respite and nepenthe from thy memories of Lenore;
Quaff, oh quaff this kind nepenthe and forget this lost Lenore!”
Quoth the Raven “Nevermore.”
“Prophet!” said I, “thing of evil!—prophet still, if bird or devil!—
Whether Tempter sent, or whether tempest tossed thee here ashore,
Desolate yet all undaunted, on this desert land enchanted—
On this home by Horror haunted—tell me truly, I implore—
Is there—is there balm in Gilead?—tell me—tell me, I implore!”
Quoth the Raven “Nevermore.”
“Prophet!” said I, “thing of evil!—prophet still, if bird or devil!
By that Heaven that bends above us—by that God we both adore—
Tell this soul with sorrow laden if, within the distant Aidenn,
It shall clasp a sainted maiden whom the angels name Lenore—
Clasp a rare and radiant maiden whom the angels name Lenore.”
Quoth the Raven “Nevermore.”
“Be that word our sign of parting, bird or fiend!” I shrieked, upstarting—
“Get thee back into the tempest and the Night’s Plutonian shore!
Leave no black plume as a token of that lie thy soul hath spoken!
Leave my loneliness unbroken!—quit the bust above my door!
Take thy beak from out my heart, and take thy form from off my door!”
Quoth the Raven “Nevermore.”
And the Raven, never flitting, still is sitting, still is sitting
On the pallid bust of Pallas just above my chamber door;
And his eyes have all the seeming of a demon’s that is dreaming,
And the lamp-light o’er him streaming throws his shadow on the floor;
And my soul from out that shadow that lies floating on the floor
Shall be lifted—nevermore!
""".strip().splitlines()
import math


def _build_toy_sequences(vocab_size: int, seq_len: int, pattern: str = "zigzag") -> torch.Tensor:
    if pattern == "constant":
        return torch.zeros(seq_len, dtype=torch.long)
    if pattern == "arange":
        return torch.arange(seq_len, dtype=torch.long) % vocab_size
    if pattern == "zigzag":
        up = torch.arange(seq_len // 2, dtype=torch.long)
        down = torch.flip(up, dims=[0])
        base = torch.cat((up, down), dim=0)
        if base.numel() < seq_len:
            base = torch.cat((base, base[: seq_len - base.numel()]), dim=0)
        return base % vocab_size
    raise ValueError(f"Unknown pattern '{pattern}'.")


class TinySequenceDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, pattern: str = "zigzag"):
        super().__init__()
        base = _build_toy_sequences(vocab_size, seq_len, pattern)
        self.samples = torch.stack([base.clone() for _ in range(num_samples)], dim=0)
        self.attention = torch.ones(num_samples, seq_len, dtype=torch.long)

    def __len__(self) -> int:
        return self.samples.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.samples[idx],
            "attention_mask": self.attention[idx],
        }


class TinyTextDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, seq_len: int, max_lines: int = 100):
        super().__init__()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        lines: List[str] = [line.strip() for line in SAMPLE_TEXT if line.strip()]
        lines = lines[:max_lines]

        encoded = tokenizer(lines, padding=False, truncation=False, add_special_tokens=True)

        chunks: List[List[int]] = []
        attn_chunks: List[List[int]] = []
        for input_ids, attn in zip(encoded["input_ids"], encoded["attention_mask"]):
            if len(input_ids) < seq_len:
                pad_len = seq_len - len(input_ids)
                padded_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                padded_attn = attn + [0] * pad_len
                chunks.append(padded_ids[:seq_len])
                attn_chunks.append(padded_attn[:seq_len])
            else:
                for idx in range(0, len(input_ids), seq_len):
                    chunk = input_ids[idx : idx + seq_len]
                    attn_chunk = attn[idx : idx + seq_len]
                    if len(chunk) < seq_len:
                        pad_len = seq_len - len(chunk)
                        chunk = chunk + [tokenizer.pad_token_id] * pad_len
                        attn_chunk = attn_chunk + [0] * pad_len
                    chunks.append(chunk[:seq_len])
                    attn_chunks.append(attn_chunk[:seq_len])

        if not chunks:
            raise ValueError("No text chunks available; reduce sequence length.")

        self.samples = torch.tensor(chunks, dtype=torch.long)
        self.attention = torch.tensor(attn_chunks, dtype=torch.long)

    def __len__(self) -> int:
        return self.samples.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.samples[idx],
            "attention_mask": self.attention[idx],
        }


@dataclass
class TrainConfig:
    vocab_size: int = 64
    embed_dim: int = 32
    depth: int = 2
    num_heads: int = 4
    ff_hidden_dim: int = 128
    seq_len: int = 32
    num_samples: int = 8
    batch_size: int = 4
    lr: float = 3e-4
    max_epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pattern: str = "zigzag"
    label_smoothing: float = 0.001
    weight_exponent: float = 1.0
    norm_epsilon: float = 1e-4
    tokenizer_name: str = "gpt2"
    use_real_text: bool = True


def train_overfit(cfg: TrainConfig) -> None:
    torch.manual_seed(42)

    if cfg.use_real_text:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        dataset: Dataset = TinyTextDataset(tokenizer, cfg.seq_len, max_lines=100)
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id or 0
    else:
        dataset = TinySequenceDataset(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            num_samples=cfg.num_samples,
            pattern=cfg.pattern,
        )
        vocab_size = cfg.vocab_size
        pad_token_id = 0

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = DLLM(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        ff_hidden_dim=cfg.ff_hidden_dim,
        pad_token_id=pad_token_id,
        label_smoothing=cfg.label_smoothing,
        weight_exponent=cfg.weight_exponent,
        norm_epsilon=cfg.norm_epsilon,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    running_loss = None
    for epoch in range(1, cfg.max_epochs + 1):
        for batch in loader:
            tokens = batch["input_ids"].to(cfg.device)
            mask = batch["attention_mask"].to(cfg.device)

            loss = model.loss(tokens, padding_mask=mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = loss.item() if running_loss is None else 0.9 * running_loss + 0.1 * loss.item()

        if epoch % 10 == 0 or epoch == cfg.max_epochs:
            print(f"[epoch {epoch:03d}] loss={loss.item():.4f} running_loss={running_loss:.4f}")

    with torch.no_grad():
        batch = next(iter(loader))
        tokens = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)

        eval_loss = model.loss(tokens, padding_mask=mask).item()
        print(f"Final training loss on toy set: {eval_loss:.4f}")

        recon_logits = model.embed_token(tokens)
        recon_tokens = recon_logits.softmax(dim=-1).argmax(dim=-1)
        token_acc = (recon_tokens == tokens).float().mean().item()
        print(f"Embed-token reconstruction accuracy: {token_acc * 100:.2f}%")

        model.eval()
        with torch.no_grad():
            state_logits = model.state_norm(recon_logits)
            velocity_pred = model.predict_velocity(state_logits, timesteps=None, padding_mask=mask)
            init_logits = model._sample_logits(recon_logits.shape, device=cfg.device, dtype=recon_logits.dtype)
            target_velocity = recon_logits - init_logits
            target_velocity = target_velocity - target_velocity.mean(dim=-1, keepdim=True)
            target_velocity = target_velocity / math.sqrt(model.vocab_size)
            velocity_mse = (velocity_pred - target_velocity).pow(2).mean().item()
        print(f"Velocity MSE at target logits: {velocity_mse:.6f}")

        sampled = model.sample(
            batch_size=1,
            seq_len=cfg.seq_len,
            steps=100,
            deterministic=True,
            device=cfg.device,
            sampler="heun",
            padding_mask=torch.ones(1, cfg.seq_len, device=cfg.device, dtype=torch.long),
        )
        sampled_ids = sampled.cpu().tolist()
        print("Sampled token ids:", sampled_ids)
        if cfg.use_real_text:
            decoded = tokenizer.batch_decode(sampled, skip_special_tokens=True)
            print("Decoded text:", decoded)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Tiny DLLM overfit test.")
    parser.add_argument("--device", default=None, help="Override device (cpu or cuda).")
    parser.add_argument("--max-epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--pattern", type=str, default=None, help="Toy sequence pattern (constant/arange/zigzag).")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer to use when real text is enabled.")
    parser.add_argument("--no-real-text", action="store_true", help="Use synthetic patterns instead of real text.")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.device:
        cfg.device = args.device
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.pattern:
        cfg.pattern = args.pattern
    if args.tokenizer_name:
        cfg.tokenizer_name = args.tokenizer_name
    if args.no_real_text:
        cfg.use_real_text = False
    return cfg


def main() -> None:
    cfg = parse_args()
    train_overfit(cfg)


if __name__ == "__main__":
    main()
