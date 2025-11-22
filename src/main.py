"""Hydra orchestrator – spawns src.train in a subprocess."""
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run_cfg_path = CONFIG_DIR / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(run_cfg_path)

    # Build composed cfg to inspect/override here (does not propagate to train)
    with open_dict(cfg):
        full_cfg = OmegaConf.merge(cfg, OmegaConf.load(run_cfg_path))
    with open_dict(full_cfg):
        if full_cfg.mode == "trial":
            full_cfg.wandb.mode = "disabled"
            full_cfg.optuna.n_trials = 0
            full_cfg.training.epochs = 1
            full_cfg.training.max_train_batches = 2
        elif full_cfg.mode == "full":
            full_cfg.wandb.mode = "online"

    # Save composed cfg for debugging
    composed_path = Path.cwd() / "composed_cfg.yaml"
    OmegaConf.save(full_cfg, composed_path)

    # Spawn training subprocess (uses hydra again inside)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    composed_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
