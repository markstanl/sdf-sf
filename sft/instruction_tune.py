"""instruction-tunes a mid-trained checkpoint on the packed set from prepare_instruction_data.py."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunked_ce import chunked_causal_lm_loss

SANITY_STEPS = 2


class PackedWindows(torch.utils.data.Dataset):
    def __init__(self, root: Path):
        self.ids = np.load(root / "input_ids.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        self.lens = json.loads((root / "doc_lens.json").read_text())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i], "doc_lens": self.lens[i]}


def collate(features):
    return {"input_ids": torch.tensor(np.stack([f["input_ids"] for f in features]), dtype=torch.long),
            "labels": torch.tensor(np.stack([f["labels"] for f in features]), dtype=torch.long),
            "position_ids": torch.stack([torch.cat([torch.arange(n) for n in f["doc_lens"]]) for f in features])}


def main():
    ap = argparse.ArgumentParser(description="Instruction-tune a mid-trained checkpoint on the packed set.")
    ap.add_argument("--config", type=Path, required=True, help="sft yaml (see configs/sft.yaml)")
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--micro-batch", type=int, default=None, help="overrides optim.micro_batch")
    ap.add_argument("--sanity", action="store_true", help=f"run {SANITY_STEPS} steps; skip the size assert")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    opt = cfg["optim"]
    steps = SANITY_STEPS if args.sanity else int(cfg["steps"])
    global_batch = int(cfg["global_batch"])
    micro_batch = args.micro_batch if args.micro_batch is not None else int(opt["micro_batch"])
    world = int(os.environ.get("WORLD_SIZE", 1))
    accumulation = global_batch // (micro_batch * world)
    assert accumulation * micro_batch * world == global_batch
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    model = AutoModelForCausalLM.from_pretrained(args.init, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.loss_function = chunked_causal_lm_loss
    train = PackedWindows(args.data)
    if not args.sanity:
        assert len(train) >= steps * global_batch
    training = TrainingArguments(
        output_dir=str(args.out), max_steps=steps, seed=int(cfg["seed"]),
        per_device_train_batch_size=micro_batch, gradient_accumulation_steps=accumulation,
        learning_rate=float(opt["learning_rate"]), lr_scheduler_type=opt["scheduler"],
        warmup_ratio=float(opt["warmup_ratio"]), weight_decay=float(opt["weight_decay"]),
        adam_beta1=float(opt["betas"][0]), adam_beta2=float(opt["betas"][1]), adam_epsilon=1e-8,
        max_grad_norm=float(opt["clip"]),
        bf16=True, logging_steps=10, save_strategy="no", report_to=[],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp="full_shard auto_wrap", fsdp_config={"transformer_layer_cls_to_wrap": [cfg["fsdp_wrap_cls"]]},
        dataloader_num_workers=2, remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=train, data_collator=collate)
    trainer.train()
    assert trainer.state.global_step == steps
    trainer.save_model(str(args.out / "final"))
    tok.save_pretrained(str(args.out / "final"))


if __name__ == "__main__":
    main()
