import math
import os
from types import MethodType
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBConditionPredictor(nn.Module):
    """6-channel RGB pair regressor.

    Input:  source RGB + target/generated RGB, concatenated as [B, 6, H, W].
    Output: [sin(theta), cos(theta), range / range_scale].
    """
    def __init__(self, model: str = "resnet18"):
        super().__init__()
        try:
            import torchvision.models as tvm
        except Exception as e:
            raise ImportError("torchvision is required for RGBConditionPredictor") from e

        if model == "resnet18":
            net = tvm.resnet18(weights=None)
        elif model == "resnet34":
            net = tvm.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported RGB predictor model: {model}")

        old_conv = net.conv1
        new_conv = nn.Conv2d(
            6,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
        net.conv1 = new_conv
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, 3)
        self.net = net

    def forward(self, source_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([source_rgb, target_rgb], dim=1)
        return self.net(x)


def _load_rgb_predictor(checkpoint: str, device: torch.device, model_name: str = "resnet18") -> Tuple[RGBConditionPredictor, Dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu")

    meta = {}
    if isinstance(ckpt, dict):
        meta = {
            "epoch": ckpt.get("epoch", None),
            "global_step": ckpt.get("global_step", None),
            "metrics": ckpt.get("metrics", None),
        }

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        k = k.replace("module.", "", 1)
        k = k.replace("rgb_condition_predictor.", "", 1)
        clean_state[k] = v

    predictor = RGBConditionPredictor(model=model_name)
    target_state = predictor.state_dict()
    target_keys = set(target_state.keys())

    candidates = [
        clean_state,
        {"net." + k: v for k, v in clean_state.items()},
        {"model." + k: v for k, v in clean_state.items()},
    ]

    best_state = clean_state
    best_hits = -1
    for cand in candidates:
        hits = 0
        for k, v in cand.items():
            if k in target_keys and target_state[k].shape == v.shape:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_state = cand

    missing, unexpected = predictor.load_state_dict(best_state, strict=False)
    print(
        f"[RGBLoss] predictor load_state_dict: "
        f"matched={best_hits}, missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )

    predictor.to(device)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    return predictor, meta

def _subset_batch(batch: Any, indices: torch.Tensor) -> Any:
    """Subset a batch while preserving dict/list/string structures."""
    if isinstance(batch, torch.Tensor):
        if batch.shape[0] == indices.shape[0]:
            return batch
        return batch.index_select(0, indices.to(batch.device))
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            out[k] = _subset_batch(v, indices) if _can_subset(v, len(indices), batch_size_hint=_infer_batch_size(batch)) else v
        return out
    if isinstance(batch, (list, tuple)):
        # List of per-sample items or list of tensors.
        try:
            bsz = len(batch)
            max_idx = int(indices.max().item()) if indices.numel() else -1
            if bsz > max_idx and not isinstance(batch[0], torch.Tensor):
                return [batch[int(i)] for i in indices.detach().cpu().tolist()]
        except Exception:
            pass
        return type(batch)(_subset_batch(v, indices) for v in batch)
    return batch


def _infer_batch_size(batch: Any) -> Optional[int]:
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                return int(v.shape[0])
            if isinstance(v, (list, tuple)):
                return len(v)
    return None


def _can_subset(v: Any, n_idx: int, batch_size_hint: Optional[int] = None) -> bool:
    if isinstance(v, torch.Tensor) and v.ndim >= 1:
        return batch_size_hint is None or int(v.shape[0]) == int(batch_size_hint)
    if isinstance(v, (list, tuple)):
        return batch_size_hint is None or len(v) == int(batch_size_hint)
    if isinstance(v, dict):
        return True
    return False


def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D image tensor, got shape={tuple(x.shape)}")
    # NHWC -> NCHW
    if x.shape[1] not in (1, 3, 4, 6) and x.shape[-1] in (1, 3, 4, 6):
        x = x.permute(0, 3, 1, 2).contiguous()
    return x


def _to_01_rgb(x: torch.Tensor) -> torch.Tensor:
    x = _to_bchw(x).float()
    if x.shape[1] > 3:
        x = x[:, :3]
    # Heuristic: target jpg in LDM is usually [-1, 1], hint is usually [0, 1].
    if float(x.detach().min()) < -0.05:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def _find_source_rgb(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    for key in ["hint", "source", "source_img", "source_image", "src", "control"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            return _to_01_rgb(batch[key].to(device))
    # fallback: use target jpg; this should not normally happen.
    if "jpg" in batch and isinstance(batch["jpg"], torch.Tensor):
        print("[RGBLoss][Warn] source image key not found; falling back to batch['jpg']", flush=True)
        return _to_01_rgb(batch["jpg"].to(device))
    raise KeyError(f"Could not find source RGB in batch. Available keys={list(batch.keys())}")


def _extract_pose_tensor(batch: Dict[str, Any], device: torch.device, range_scale: float) -> torch.Tensor:
    """Return target [sin(theta), cos(theta), range/range_scale].

    Supports either:
      - raw [heading_deg, range]
      - already [sin, cos, range_scaled] if tensor has >=3 dims and first two are in [-1,1].
      - separate heading/range keys.
    """
    candidate_keys = [
        "num_cond", "numeric_cond", "pose", "condition", "cond", "relative_pose",
        "pred_pose", "true_pose", "meta_cond", "label",
    ]
    for key in candidate_keys:
        v = batch.get(key, None)
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] >= 2:
            v = v.to(device).float()
            if v.shape[1] >= 3:
                first_two = v[:, :2]
                # likely already sin/cos if values are in [-1.2, 1.2]
                if float(first_two.detach().abs().max()) <= 1.2:
                    out = torch.stack([v[:, 0], v[:, 1], v[:, 2]], dim=1)
                    return out
            heading = v[:, 0]
            rag = v[:, 1]
            rad = heading * math.pi / 180.0
            return torch.stack([torch.sin(rad), torch.cos(rad), rag / float(range_scale)], dim=1)

    heading_key = None
    range_key = None
    for k in ["pred_deg_num", "true_deg_num", "heading_num", "heading", "deg", "angle", "theta"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            heading_key = k
            break
    for k in ["pred_rag_num", "true_rag_num", "range_num", "range", "rag", "distance", "dist"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            range_key = k
            break
    if heading_key is not None and range_key is not None:
        heading = batch[heading_key].to(device).float().view(-1)
        rag = batch[range_key].to(device).float().view(-1)
        rad = heading * math.pi / 180.0
        return torch.stack([torch.sin(rad), torch.cos(rad), rag / float(range_scale)], dim=1)

    raise KeyError(f"Could not find pose keys for RGB loss. Available keys={list(batch.keys())}")


def _subset_cond(c: Any, idx: torch.Tensor) -> Any:
    if isinstance(c, torch.Tensor):
        return c.index_select(0, idx.to(c.device)) if c.ndim >= 1 and c.shape[0] >= int(idx.max().item()) + 1 else c
    if isinstance(c, list):
        return [_subset_cond(v, idx) for v in c]
    if isinstance(c, tuple):
        return tuple(_subset_cond(v, idx) for v in c)
    if isinstance(c, dict):
        return {k: _subset_cond(v, idx) for k, v in c.items()}
    return c


def _predict_x0_from_model(pl_module, x_start: torch.Tensor, cond: Any, t_min: int, t_max: int) -> torch.Tensor:
    b = x_start.shape[0]
    device = x_start.device
    num_timesteps = int(getattr(pl_module, "num_timesteps", 1000))
    lo = max(0, int(t_min))
    hi = int(t_max) if int(t_max) > 0 else num_timesteps - 1
    hi = min(hi, num_timesteps - 1)
    if hi < lo:
        hi = lo
    t = torch.randint(lo, hi + 1, (b,), device=device).long()
    noise = torch.randn_like(x_start)
    x_noisy = pl_module.q_sample(x_start=x_start, t=t, noise=noise)
    model_output = pl_module.apply_model(x_noisy, t, cond)
    parameterization = getattr(pl_module, "parameterization", "eps")
    if parameterization == "eps":
        x0 = pl_module.predict_start_from_noise(x_noisy, t=t, noise=model_output)
    elif parameterization == "x0":
        x0 = model_output
    elif parameterization == "v" and hasattr(pl_module, "predict_start_from_z_and_v"):
        x0 = pl_module.predict_start_from_z_and_v(x_noisy, t=t, v=model_output)
    else:
        raise NotImplementedError(f"Unsupported parameterization for RGB loss: {parameterization}")
    return x0



def _save_rgb_loss_curve(history, save_dir: str, step: int):
    """Save a lightweight PNG loss curve for quick training inspection."""
    if not history:
        return
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "rgb_loss_history.csv")
    latest_png = os.path.join(save_dir, "rgb_loss_latest.png")
    step_png = os.path.join(save_dir, f"rgb_loss_step_{int(step):06d}.png")

    # Always write a small CSV so the curve can be replotted later.
    keys = ["step", "base", "rgb", "weighted_rgb", "total", "angle", "range", "norm"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["step"] for r in history]
        base = [r["base"] for r in history]
        rgb = [r["rgb"] for r in history]
        weighted = [r["weighted_rgb"] for r in history]
        angle = [r["angle"] for r in history]
        rng = [r["range"] for r in history]

        plt.figure(figsize=(9, 5), dpi=140)
        plt.plot(xs, base, label="base diffusion")
        plt.plot(xs, rgb, label="rgb cond raw")
        plt.plot(xs, weighted, label="weighted rgb")
        plt.plot(xs, angle, label="angle")
        plt.plot(xs, rng, label="range")
        plt.xlabel("global step")
        plt.ylabel("loss")
        plt.title(f"RGB condition loss monitor @ step {int(step)}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(latest_png)
        plt.savefig(step_png)
        plt.close()
        print(f"[RGBLossPlot] saved: {latest_png} and {step_png}", flush=True)
    except Exception as e:
        print(f"[RGBLossPlot][Warn] failed to save plot: {repr(e)}", flush=True)

def patch_model_with_rgb_condition_loss(model, args, image_size: int):
    """Monkey-patch a Lightning model's training_step to add frozen RGB condition loss.

    This is intentionally isolated from cldm/model.py so the baseline code remains untouched.
    """
    if not getattr(args, "rgb_loss_checkpoint", None):
        print("[RGBLoss] disabled", flush=True)
        return model

    device = next(model.parameters()).device
    predictor, meta = _load_rgb_predictor(args.rgb_loss_checkpoint, device=device, model_name=args.rgb_loss_model)
    model.rgb_condition_predictor = predictor
    model.rgb_loss_args = args
    model.rgb_loss_image_size = int(getattr(args, "rgb_loss_image_size", 0) or image_size)
    model._rgb_original_training_step = model.training_step
    model._rgb_loss_history = []
    model._rgb_loss_plot_dir = (
        getattr(args, "rgb_loss_plot_dir", None)
        or os.path.join(getattr(args, "output_dir", "."), "rgb_loss_curves")
    )

    print("\n========== RGB condition loss ==========")
    print(f"checkpoint:     {args.rgb_loss_checkpoint}")
    print(f"weight:         {args.rgb_loss_weight}")
    print(f"start_step:     {args.rgb_loss_start_step}")
    print(f"every_n_steps:  {args.rgb_loss_every_n_steps}")
    print(f"loss_batch:     {args.rgb_loss_batch_size}")
    print(f"range_scale:    {args.rgb_loss_range_scale}")
    print(f"t range:        [{args.rgb_loss_t_min}, {args.rgb_loss_t_max}]")
    print("========================================\n")

    def training_step_with_rgb_loss(self, batch, batch_idx):
        base_out = self._rgb_original_training_step(batch, batch_idx)

        # Parse base loss return.
        if isinstance(base_out, dict):
            base_loss = base_out.get("loss", None)
            if base_loss is None:
                return base_out
        else:
            base_loss = base_out

        a = self.rgb_loss_args
        step = int(getattr(self, "global_step", 0))
        if a.rgb_loss_weight <= 0 or step < int(a.rgb_loss_start_step):
            return base_out
        if int(a.rgb_loss_every_n_steps) > 1 and (step % int(a.rgb_loss_every_n_steps) != 0):
            return base_out
        if not isinstance(batch, dict):
            if step == int(a.rgb_loss_start_step):
                print("[RGBLoss][Warn] batch is not a dict; skip RGB loss", flush=True)
            return base_out

        try:
            bsz = _infer_batch_size(batch)
            if bsz is None or bsz <= 0:
                return base_out
            m = min(int(a.rgb_loss_batch_size), int(bsz)) if int(a.rgb_loss_batch_size) > 0 else int(bsz)
            # Random subset to keep overhead controlled.
            idx = torch.randperm(int(bsz), device=base_loss.device)[:m]
            small_batch = _subset_batch(batch, idx.detach().cpu())

            x_start, cond = self.get_input(small_batch, self.first_stage_key)
            x_start = x_start.to(base_loss.device)
            cond = _subset_cond(cond, torch.arange(x_start.shape[0], device=base_loss.device))

            x0 = _predict_x0_from_model(
                self,
                x_start=x_start,
                cond=cond,
                t_min=int(a.rgb_loss_t_min),
                t_max=int(a.rgb_loss_t_max),
            )
            gen_rgb = self.decode_first_stage(x0)
            gen_rgb = _to_01_rgb(gen_rgb)
            src_rgb = _find_source_rgb(small_batch, base_loss.device)
            target_pose = _extract_pose_tensor(small_batch, base_loss.device, range_scale=float(a.rgb_loss_range_scale))

            # Match predictor input resolution.
            pred_size = int(getattr(self, "rgb_loss_image_size", gen_rgb.shape[-1]))
            if src_rgb.shape[-1] != pred_size or src_rgb.shape[-2] != pred_size:
                src_rgb = F.interpolate(src_rgb, size=(pred_size, pred_size), mode="bilinear", align_corners=False)
            if gen_rgb.shape[-1] != pred_size or gen_rgb.shape[-2] != pred_size:
                gen_rgb = F.interpolate(gen_rgb, size=(pred_size, pred_size), mode="bilinear", align_corners=False)

            pred_pose = self.rgb_condition_predictor(src_rgb, gen_rgb)
            angle_loss = F.mse_loss(pred_pose[:, :2], target_pose[:, :2])
            range_loss = F.smooth_l1_loss(pred_pose[:, 2], target_pose[:, 2])
            norm_loss = ((torch.sqrt(torch.clamp((pred_pose[:, :2] ** 2).sum(dim=1), min=1e-8)) - 1.0) ** 2).mean()
            rgb_loss = (
                float(a.rgb_loss_angle_weight) * angle_loss
                + float(a.rgb_loss_range_weight) * range_loss
                + float(a.rgb_loss_norm_weight) * norm_loss
            )
            total_loss = base_loss + float(a.rgb_loss_weight) * rgb_loss

            # Keep a tiny loss history for quick PNG monitoring.
            try:
                self._rgb_loss_history.append({
                    "step": int(step),
                    "base": float(base_loss.detach().cpu()),
                    "rgb": float(rgb_loss.detach().cpu()),
                    "weighted_rgb": float((float(a.rgb_loss_weight) * rgb_loss).detach().cpu()),
                    "total": float(total_loss.detach().cpu()),
                    "angle": float(angle_loss.detach().cpu()),
                    "range": float(range_loss.detach().cpu()),
                    "norm": float(norm_loss.detach().cpu()),
                })
            except Exception:
                pass

            # Lightning logging; keep it lightweight.
            try:
                self.log("train/rgb_cond_loss", rgb_loss.detach(), prog_bar=False, logger=True, on_step=True, on_epoch=False)
                self.log("train/rgb_angle_loss", angle_loss.detach(), prog_bar=False, logger=True, on_step=True, on_epoch=False)
                self.log("train/rgb_range_loss", range_loss.detach(), prog_bar=False, logger=True, on_step=True, on_epoch=False)
            except Exception:
                pass

            if step % int(max(a.rgb_loss_print_every, 1)) == 0:
                print(
                    f"[RGBLoss] step={step} total={float(total_loss.detach()):.5f} "
                    f"base={float(base_loss.detach()):.5f} rgb={float(rgb_loss.detach()):.5f} "
                    f"angle={float(angle_loss.detach()):.5f} range={float(range_loss.detach()):.5f} norm={float(norm_loss.detach()):.5f}",
                    flush=True,
                )

            plot_every = int(getattr(a, "rgb_loss_plot_every", 0) or 0)
            if plot_every > 0 and step > 0 and step % plot_every == 0:
                _save_rgb_loss_curve(
                    getattr(self, "_rgb_loss_history", []),
                    getattr(self, "_rgb_loss_plot_dir", "rgb_loss_curves"),
                    step,
                )

            if isinstance(base_out, dict):
                out = dict(base_out)
                out["loss"] = total_loss
                return out
            return total_loss
        except Exception as e:
            # Do not crash a long training job because RGB auxiliary loss failed once.
            print(f"[RGBLoss][ERROR] skipped due to: {repr(e)}", flush=True)
            return base_out

    model.training_step = MethodType(training_step_with_rgb_loss, model)
    return model
