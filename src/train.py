import os
import json
import random
import sys
from pathlib import Path
from typing import Tuple, Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from src.preprocess import build_dataloaders, get_tokenizer
from src.model import (
    build_optimizer_with_layerwise_scaling,
    evaluate_accuracy,
    load_lm_and_prepare_for_training,
    save_lora_adapters,
)

# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def _set_seed(seed: int = 42):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------------------------------------------------------
# Mode-specific overrides
# -----------------------------------------------------------------------------

def _apply_mode_overrides(cfg: DictConfig) -> None:
    """Mutates cfg according to cfg.mode (trial/full)."""
    with open_dict(cfg):
        if cfg.mode == "trial":
            cfg.wandb.mode = "disabled"
            cfg.optuna.n_trials = 0
            cfg.training.epochs = 1
            cfg.training.max_train_batches = 2
        elif cfg.mode == "full":
            cfg.wandb.mode = "online"
        else:
            raise ValueError(f"Unknown execution mode '{cfg.mode}'. Allowed: trial | full")


# -----------------------------------------------------------------------------
# Weights & Biases helpers
# -----------------------------------------------------------------------------

def _maybe_init_wandb(cfg: DictConfig):
    """Initialise wandb unless disabled. Returns wandb run or None."""
    if cfg.wandb.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return None

    import wandb  # local import to avoid mandatory wandb dependency in trial

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run_id,
        resume="allow",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"[wandb] Run url: {run.get_url()}")
    return run


# -----------------------------------------------------------------------------
# Single training pass (no hyper-parameter search)
# -----------------------------------------------------------------------------

def _train_once(cfg: DictConfig) -> Tuple[float, float]:
    """Train a model once with fixed hyper-parameters.
    Returns (best_val_acc, final_test_acc).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(cfg)
    model = load_lm_and_prepare_for_training(cfg, device)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, tokenizer)

    optimizer, scheduler, _ = build_optimizer_with_layerwise_scaling(cfg, model, len(train_loader))

    wandb_run = _maybe_init_wandb(cfg)

    gradient_accum = cfg.training.gradient_accumulation_steps
    global_step = 0
    best_val_acc, best_epoch = 0.0, 0

    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
            if cfg.training.max_train_batches and batch_idx >= cfg.training.max_train_batches:
                break

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / gradient_accum
            loss.backward()
            running_loss += loss.item() * gradient_accum

            if (batch_idx + 1) % gradient_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if wandb_run:
                    wandb_run.log({"train_loss": running_loss / (batch_idx + 1)}, step=global_step)

            pbar.set_postfix({"loss": f"{running_loss / (batch_idx + 1):.4f}"})

        # ---------------- Validation & Test ----------------
        val_acc, _, _ = evaluate_accuracy(model, tokenizer, val_loader, device, cfg)
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch

        test_acc, test_preds, test_golds = evaluate_accuracy(model, tokenizer, test_loader, device, cfg)

        if wandb_run:
            wandb_run.log({"val_accuracy": val_acc, "test_accuracy": test_acc}, step=global_step)

        print(f"Epoch {epoch}: val_acc={val_acc:.4f} | test_acc={test_acc:.4f}")

    # ---------------- Summary ----------------
    if wandb_run:
        wandb_run.summary["best_val_accuracy"] = best_val_acc
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["test_accuracy"] = test_acc
        wandb_run.summary["_preds"] = test_preds  # For evaluation script
        wandb_run.summary["_golds"] = test_golds
        wandb_run.finish()

    # save adapters when meaningful
    if cfg.mode == "full" and cfg.get("results_dir"):
        save_dir = Path(cfg.results_dir) / cfg.run_id
        save_lora_adapters(model, save_dir)

    return best_val_acc, test_acc


# -----------------------------------------------------------------------------
# Optuna hyper-parameter search (outer run only logs best to wandb)
# -----------------------------------------------------------------------------

def _optuna_search(cfg: DictConfig) -> Dict[str, Any]:
    import optuna

    def objective(trial: optuna.Trial):
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        # --- sample hyper-parameters defined in cfg.optuna.search_space
        for hp_name, hp_cfg in cfg.optuna.search_space.items():
            if hp_cfg.type == "loguniform":
                sampled = trial.suggest_float(hp_name, hp_cfg.low, hp_cfg.high, log=True)
            elif hp_cfg.type == "uniform":
                sampled = trial.suggest_float(hp_name, hp_cfg.low, hp_cfg.high, log=False)
            else:
                raise ValueError(f"Unsupported hp type {hp_cfg.type}")

            if hp_name == "beta":
                trial_cfg.training.layer_lr_scaling.beta = sampled
            elif hp_name == "learning_rate":
                trial_cfg.training.optimizer.learning_rate = sampled
            elif hp_name == "weight_decay":
                trial_cfg.training.optimizer.weight_decay = sampled
            else:
                OmegaConf.update(trial_cfg, hp_name, sampled)

        trial_cfg.wandb.mode = "disabled"  # disable wandb inside trials
        best_val, _ = _train_once(trial_cfg)
        return best_val

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=cfg.optuna.n_trials)

    print("[Optuna] Best value:", study.best_value)
    print("[Optuna] Best params:", study.best_trial.params)
    return study.best_trial.params


# -----------------------------------------------------------------------------
# Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Merge root config with run-specific YAML
    run_cfg_path = Path(__file__).resolve().parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(run_cfg_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_cfg_path))

    _apply_mode_overrides(cfg)
    _set_seed(42)

    # Hyper-parameter optimisation (optional)
    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0:
        best_params = _optuna_search(cfg)
        with open_dict(cfg):
            for k, v in best_params.items():
                if k == "beta":
                    cfg.training.layer_lr_scaling.beta = v
                elif k == "learning_rate":
                    cfg.training.optimizer.learning_rate = v
                elif k == "weight_decay":
                    cfg.training.optimizer.weight_decay = v

    # Final training with (possibly) tuned params
    _train_once(cfg)


if __name__ == "__main__":
    main()
