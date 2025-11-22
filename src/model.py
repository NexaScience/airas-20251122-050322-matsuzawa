"""Model utilities – loading, optimisation, evaluation, adapter saving."""
from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    # Graceful degradation: training without LoRA if peft unavailable
    LoraConfig = None

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_lm_and_prepare_for_training(cfg, device: torch.device):
    """Load a language model according to cfg, optionally with 4-bit quantisation & LoRA.
    Falls back to fp16/bf16 on CPU when 4-bit is requested but CUDA unavailable.
    """
    use_4bit = cfg.model.precision == "4bit" and torch.cuda.is_available()

    if cfg.model.precision == "4bit" and not torch.cuda.is_available():
        print("[WARN] 4-bit quantisation requested but CUDA not available – falling back to fp16.")

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        cache_dir=str(CACHE_DIR),
        quantization_config=quant_cfg,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # ---------------- LoRA adapters ----------------
    if cfg.model.adapter.type == "lora":
        if LoraConfig is None:
            raise ImportError("peft package required for LoRA adapters but not found.")
        lora_cfg = LoraConfig(
            r=cfg.model.adapter.r,
            lora_alpha=cfg.model.adapter.alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=cfg.model.adapter.dropout,
            task_type="CAUSAL_LM",
        )
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.to(device)
    return model

# -----------------------------------------------------------------------------
# Optimiser & scheduler with ALLRS
# -----------------------------------------------------------------------------

def _layer_id_from_name(name: str) -> int:
    parts = name.split(".")
    for p in parts:
        if p.isdigit():
            return int(p)
    return -1


def _compute_layer_lr_scale(name: str, num_layers: int, beta: float) -> float:
    if any(tok in name for tok in [".layers.", "transformer.h."]):
        idx = _layer_id_from_name(name)
        if idx >= 0:
            return beta ** (num_layers - 1 - idx)
    return 1.0


def build_optimizer_with_layerwise_scaling(cfg, model, train_loader_len: int):
    base_lr = cfg.training.optimizer.learning_rate
    wd = cfg.training.optimizer.weight_decay
    beta = cfg.training.layer_lr_scaling.beta if cfg.training.layer_lr_scaling.enabled else 1.0

    # attempt to infer number of layers from model config
    num_layers = (
        getattr(model.config, "num_hidden_layers", None)
        or getattr(model.config, "n_layer", None)
        or getattr(model.config, "num_layers", 0)
    )

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scale = _compute_layer_lr_scale(name, num_layers, beta)
        lr = base_lr * scale
        key = f"lr{lr}_wd{wd}"
        if key not in param_groups:
            param_groups[key] = {"params": [], "lr": lr, "weight_decay": wd}
        param_groups[key]["params"].append(param)

    optimizer = torch.optim.AdamW(list(param_groups.values()), betas=tuple(cfg.training.optimizer.betas), eps=cfg.training.optimizer.eps)

    steps_per_epoch = train_loader_len // cfg.training.gradient_accumulation_steps
    if cfg.training.max_train_batches:
        steps_per_epoch = min(steps_per_epoch, cfg.training.max_train_batches // cfg.training.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * cfg.training.epochs)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.scheduler.warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler, total_steps

# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def evaluate_accuracy(model, tokenizer, loader, device, cfg) -> Tuple[float, list, list]:
    model.eval()
    preds, golds = [], []
    correct = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
            )
            input_lens = (input_ids != tokenizer.pad_token_id).sum(-1)
            for i in range(outputs.size(0)):
                gen_ids = outputs[i][input_lens[i] :]
                pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
                gold = batch["labels"][i]
                preds.append(pred.strip())
                golds.append(gold.strip())
                if pred.strip().split("####")[-1].strip() == gold.strip():
                    correct += 1
            if cfg.mode == "trial" and batch_idx >= 20:
                break
    total = len(preds) if preds else 1
    acc = correct / total
    model.train()
    return acc, preds, golds

# -----------------------------------------------------------------------------
# Adapter saver
# -----------------------------------------------------------------------------

def save_lora_adapters(model, out_dir: Path):
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
    except Exception as e:
        print(f"[WARN] Saving adapters failed: {e}")
