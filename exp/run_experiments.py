#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

ENV_KEY_MAP = {
    "data_path": "DATA_PATH",
    "tokenizer_path": "TOKENIZER_PATH",
    "vocab_size": "VOCAB_SIZE",
    "seed": "SEED",
    "val_batch_size": "VAL_BATCH_SIZE",
    "val_loss_every": "VAL_LOSS_EVERY",
    "train_log_every": "TRAIN_LOG_EVERY",
    "iterations": "ITERATIONS",
    "warmdown_iters": "WARMDOWN_ITERS",
    "warmup_steps": "WARMUP_STEPS",
    "train_batch_tokens": "TRAIN_BATCH_TOKENS",
    "train_seq_len": "TRAIN_SEQ_LEN",
    "max_wallclock_seconds": "MAX_WALLCLOCK_SECONDS",
    "qk_gain_init": "QK_GAIN_INIT",
    "num_layers": "NUM_LAYERS",
    "num_kv_heads": "NUM_KV_HEADS",
    "model_dim": "MODEL_DIM",
    "num_heads": "NUM_HEADS",
    "mlp_mult": "MLP_MULT",
    "tie_embeddings": "TIE_EMBEDDINGS",
    "rope_base": "ROPE_BASE",
    "logit_softcap": "LOGIT_SOFTCAP",
    "embed_lr": "EMBED_LR",
    "head_lr": "HEAD_LR",
    "tied_embed_lr": "TIED_EMBED_LR",
    "tied_embed_init_std": "TIED_EMBED_INIT_STD",
    "matrix_lr": "MATRIX_LR",
    "scalar_lr": "SCALAR_LR",
    "muon_momentum": "MUON_MOMENTUM",
    "muon_backend_steps": "MUON_BACKEND_STEPS",
    "muon_momentum_warmup_start": "MUON_MOMENTUM_WARMUP_START",
    "muon_momentum_warmup_steps": "MUON_MOMENTUM_WARMUP_STEPS",
    "beta1": "BETA1",
    "beta2": "BETA2",
    "adam_eps": "ADAM_EPS",
    "grad_clip_norm": "GRAD_CLIP_NORM",
    "enable_wandb": "ENABLE_WANDB",
    "wandb_project": "WANDB_PROJECT",
    "wandb_entity": "WANDB_ENTITY",
    "wandb_run_name": "WANDB_RUN_NAME",
    "wandb_group": "WANDB_GROUP",
    "wandb_tags": "WANDB_TAGS",
    "wandb_notes": "WANDB_NOTES",
    "wandb_mode": "WANDB_MODE",
    "wandb_dir": "WANDB_DIR",
    "wandb_upload_artifacts": "WANDB_UPLOAD_ARTIFACTS",
    "wandb_api_key": "WANDB_API_KEY",
}

MODEL_KEYS = ("num_layers", "model_dim", "num_heads", "num_kv_heads", "mlp_mult", "vocab_size")
CONTROL_MODES = {"fixed_tokens", "fixed_model", "fixed_compute"}
DEFAULT_TRAINER_PATH = "exp/baseline-sp1024/train_gpt.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trainer experiments on 8 GPUs.")
    parser.add_argument("manifest", type=Path, help="Path to experiment manifest JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved runs without launching training")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where batch outputs are written (default: <manifest_dir>/logs)",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    return manifest


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def merge_sections(defaults: dict[str, Any], experiment: dict[str, Any], key: str) -> dict[str, Any]:
    merged = dict(require_mapping(defaults.get(key), f"defaults.{key}"))
    merged.update(require_mapping(experiment.get(key), f"experiments[].{key}"))
    return merged


def merge_named_mappings(*sections: tuple[Any, str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value, name in sections:
        merged.update(require_mapping(value, name))
    return merged


def normalize_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "-")
    return "".join(out).strip("-") or "experiment"


def validate_common_config(config: dict[str, Any]) -> None:
    train_batch_tokens = int(config["train_batch_tokens"])
    train_seq_len = int(config["train_seq_len"])
    if train_batch_tokens % train_seq_len != 0:
        raise ValueError(
            f"TRAIN_BATCH_TOKENS ({train_batch_tokens}) must be divisible by TRAIN_SEQ_LEN ({train_seq_len})"
        )
    if int(config["num_heads"]) <= 0 or int(config["num_kv_heads"]) <= 0:
        raise ValueError("num_heads and num_kv_heads must be positive")
    if int(config["model_dim"]) % int(config["num_heads"]) != 0:
        raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
    if int(config["num_heads"]) % int(config["num_kv_heads"]) != 0:
        raise ValueError("NUM_HEADS must be divisible by NUM_KV_HEADS")


def resolve_control(config: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    mode = control.get("mode")
    if mode not in CONTROL_MODES:
        raise ValueError(f"Unsupported control mode: {mode}")

    resolved = {
        "mode": mode,
        "target_train_tokens": 0,
        "target_wallclock_seconds": 0.0,
    }

    if mode in {"fixed_tokens", "fixed_model"}:
        if "target_train_tokens" not in control:
            raise ValueError(f"control.target_train_tokens is required for mode={mode}")
        target_train_tokens = int(control["target_train_tokens"])
        if target_train_tokens <= 0:
            raise ValueError("control.target_train_tokens must be positive")
        train_batch_tokens = int(config["train_batch_tokens"])
        config["iterations"] = max(1, math.ceil(target_train_tokens / train_batch_tokens))
        config.setdefault("max_wallclock_seconds", 0.0)
        resolved["target_train_tokens"] = target_train_tokens
        resolved["actual_train_tokens"] = int(config["iterations"]) * train_batch_tokens
    else:
        if "target_wallclock_seconds" not in control:
            raise ValueError("control.target_wallclock_seconds is required for mode=fixed_compute")
        target_wallclock_seconds = float(control["target_wallclock_seconds"])
        if target_wallclock_seconds <= 0:
            raise ValueError("control.target_wallclock_seconds must be positive")
        resolved["target_wallclock_seconds"] = target_wallclock_seconds
        config["max_wallclock_seconds"] = target_wallclock_seconds
        config["iterations"] = int(control.get("iterations_cap", config.get("iterations", 1_000_000)))
        if int(config["iterations"]) <= 0:
            raise ValueError("iterations must be positive")
        resolved["actual_train_tokens"] = None

    if int(config["iterations"]) <= 0:
        raise ValueError("ITERATIONS must be positive after control resolution")
    return resolved


def strip_named_fields(value: dict[str, Any], name: str) -> dict[str, Any]:
    payload = dict(require_mapping(value, name))
    for key in ("name", "id", "data", "model", "overrides", "trainer", "trainer_path"):
        payload.pop(key, None)
    return payload


def expand_manifest_experiments(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_experiments = manifest.get("experiments")
    if explicit_experiments is not None:
        experiments = require_list(explicit_experiments, "experiments")
        if not experiments:
            raise ValueError("Manifest must contain a non-empty experiments array")
        expanded: list[dict[str, Any]] = []
        for experiment in experiments:
            if not isinstance(experiment, dict):
                raise ValueError("Each experiments[] entry must be an object")
            expanded.append(deepcopy(experiment))
        return expanded

    variants = require_list(manifest.get("variants"), "variants")
    controls = require_list(manifest.get("controls"), "controls")
    if not variants or not controls:
        raise ValueError("Manifest must provide either experiments[] or both variants[] and controls[]")

    expanded = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("Each variants[] entry must be an object")
        variant_name = str(variant.get("name") or variant.get("id") or "")
        if not variant_name:
            raise ValueError("Each variants[] entry must define name")
        trainer_path = variant.get("trainer_path") or variant.get("trainer") or manifest.get("trainer_path") or manifest.get("trainer")
        trainer_path = str(trainer_path or DEFAULT_TRAINER_PATH)

        for control in controls:
            if not isinstance(control, dict):
                raise ValueError("Each controls[] entry must be an object")
            control_name = str(control.get("name") or control.get("id") or "")
            if not control_name:
                raise ValueError("Each controls[] entry must define name")

            expanded.append(
                {
                    "name": f"{variant_name}-{control_name}",
                    "variant_name": variant_name,
                    "control_name": control_name,
                    "trainer_path": trainer_path,
                    "data": merge_named_mappings(
                        (variant.get("data"), f"variants[{variant_name!r}].data"),
                        (control.get("data"), f"controls[{control_name!r}].data"),
                    ),
                    "model": merge_named_mappings(
                        (variant.get("model"), f"variants[{variant_name!r}].model"),
                        (control.get("model"), f"controls[{control_name!r}].model"),
                    ),
                    "overrides": merge_named_mappings(
                        (variant.get("overrides"), f"variants[{variant_name!r}].overrides"),
                        (control.get("overrides"), f"controls[{control_name!r}].overrides"),
                    ),
                    "control": strip_named_fields(control, f"controls[{control_name!r}]") ,
                }
            )
    return expanded


def resolve_trainer_path(manifest_path: Path, trainer_value: str) -> Path:
    raw = Path(trainer_value)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = []
        for base in (Path.cwd(), manifest_path.parent, manifest_path.parent.parent, Path(__file__).resolve().parent.parent):
            candidate = (base / raw).resolve()
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise ValueError(f"Trainer path {trainer_value!r} not found. Tried: {tried}")


def build_run_config(
    manifest: dict[str, Any],
    experiment: dict[str, Any],
    batch_id: str,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    defaults = require_mapping(manifest.get("defaults"), "defaults")
    launcher = require_mapping(manifest.get("launcher"), "launcher")
    control = require_mapping(experiment.get("control"), "experiments[].control")
    if not control:
        raise ValueError("Each experiment must include a control object")

    config = dict(require_mapping(defaults, "defaults"))
    config.update(merge_sections(defaults, experiment, "data"))
    config.update(merge_sections(defaults, experiment, "model"))
    config.update(require_mapping(experiment.get("overrides"), "experiments[].overrides"))

    missing = [key for key in ("data_path", "tokenizer_path", "train_batch_tokens", "train_seq_len", *MODEL_KEYS) if key not in config]
    if missing:
        raise ValueError(f"Experiment {experiment.get('name', '<unnamed>')} is missing required keys: {missing}")

    validate_common_config(config)
    resolved_control = resolve_control(config, control)

    trainer_value = str(
        experiment.get("trainer_path")
        or experiment.get("trainer")
        or manifest.get("trainer_path")
        or manifest.get("trainer")
        or DEFAULT_TRAINER_PATH
    )
    trainer_path = resolve_trainer_path(manifest_path, trainer_value)

    variant_name = str(experiment.get("variant_name") or experiment.get("variant") or trainer_path.parent.name)
    control_name = str(experiment.get("control_name") or experiment.get("control_label") or control.get("mode") or "run")
    experiment_name = str(experiment.get("name") or f"{variant_name}-{control_name}")

    safe_name = sanitize_name(experiment_name)
    run_id = f"{batch_id}-{safe_name}"
    run_dir = output_root / batch_id / safe_name
    result_json_path = run_dir / "result.json"

    env = {key: value for key, value in os.environ.items()}
    for key, value in config.items():
        env_key = ENV_KEY_MAP.get(key)
        if env_key is not None and value is not None:
            env[env_key] = normalize_scalar(value)
    env["RUN_ID"] = run_id
    env["OUTPUT_DIR"] = str(run_dir)
    env["EXPERIMENT_NAME"] = experiment_name
    env["CONTROL_MODE"] = str(resolved_control["mode"])
    env["TARGET_TRAIN_TOKENS"] = str(resolved_control.get("target_train_tokens", 0) or 0)
    env["WANDB_RUN_NAME"] = run_id
    env.setdefault("WANDB_GROUP", batch_id)

    nproc_per_node = int(launcher.get("nproc_per_node", 8))
    master_port_base = int(launcher.get("master_port_base", 29500))
    experiment_index = int(experiment.get("index", 0))
    master_port = master_port_base + experiment_index

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        str(trainer_path),
    ]

    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "experiment_name": experiment_name,
        "variant_name": variant_name,
        "control_name": control_name,
        "trainer_path": str(trainer_path),
        "run_id": run_id,
        "run_dir": run_dir,
        "result_json": result_json_path,
        "resolved_config": config,
        "resolved_control": resolved_control,
        "command": command,
        "env": env,
        "nproc_per_node": nproc_per_node,
        "master_port": master_port,
    }


def build_batch_results(manifest_path: Path, batch_id: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 2,
        "kind": "experiment_batch",
        "batch_id": batch_id,
        "manifest_path": str(manifest_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
    }


def print_dry_run(run: dict[str, Any]) -> None:
    control = run["resolved_control"]
    config = run["resolved_config"]
    print(f"[{run['experiment_name']}]", flush=True)
    print(f"  variant: {run['variant_name']}", flush=True)
    print(f"  control: {run['control_name']} ({control['mode']})", flush=True)
    print(f"  trainer: {run['trainer_path']}", flush=True)
    print(f"  run_id: {run['run_id']}", flush=True)
    print(
        "  model: "
        f"layers={config['num_layers']} dim={config['model_dim']} heads={config['num_heads']} "
        f"kv_heads={config['num_kv_heads']} mlp_mult={config['mlp_mult']}",
        flush=True,
    )
    print(
        f"  tokens: target={control.get('target_train_tokens', 0)} "
        f"actual={control.get('actual_train_tokens')}",
        flush=True,
    )
    print(
        f"  wallclock: target={control.get('target_wallclock_seconds', 0.0)} "
        f"max={config.get('max_wallclock_seconds')}",
        flush=True,
    )
    print(f"  output: {run['result_json']}", flush=True)
    print(f"  command: {' '.join(run['command'])}", flush=True)


def run_experiment(run: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(command := run["command"], env=run["env"], check=False)
    result: dict[str, Any] = {
        "experiment_name": run["experiment_name"],
        "variant_name": run["variant_name"],
        "control_name": run["control_name"],
        "trainer_path": run["trainer_path"],
        "run_id": run["run_id"],
        "command": command,
        "returncode": int(completed.returncode),
        "result_json": str(run["result_json"]),
        "resolved_control": run["resolved_control"],
        "resolved_config": run["resolved_config"],
        "status": "failed",
    }
    if completed.returncode == 0 and run["result_json"].exists():
        with run["result_json"].open("r", encoding="utf-8") as f:
            result_payload = json.load(f)
        result["status"] = "succeeded"
        result["result"] = result_payload
    elif completed.returncode == 0:
        result["error"] = f"Trainer finished without writing {run['result_json'].name}"
    else:
        result["error"] = f"Training exited with return code {completed.returncode}"
    return result


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    experiments = expand_manifest_experiments(manifest)

    batch_id = datetime.now().strftime("experiment-%Y%m%d-%H%M%S")
    output_root = (args.output_root or manifest_path.parent / "logs").resolve()
    runs_to_execute = []
    for index, experiment in enumerate(experiments):
        experiment_with_index = deepcopy(experiment)
        experiment_with_index["index"] = index
        runs_to_execute.append(build_run_config(manifest, experiment_with_index, batch_id, output_root, manifest_path))

    if args.dry_run:
        for run in runs_to_execute:
            print_dry_run(run)
        return 0

    results = []
    for run in runs_to_execute:
        print(
            f"Running {run['variant_name']} / {run['control_name']} "
            f"({run['run_id']})",
            flush=True,
        )
        results.append(run_experiment(run))

    batch_dir = output_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    results_path = batch_dir / "results.json"
    results_path.write_text(
        json.dumps(build_batch_results(manifest_path, batch_id, results), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote batch results to {results_path}", flush=True)
    return 0 if all(result["status"] == "succeeded" for result in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
