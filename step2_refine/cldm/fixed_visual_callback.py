import os
import json

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from cldm.test_callback import TestDatasetWithPrediction
from ldm.models.diffusion.ddim import DDIMSampler
try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except Exception:
    try:
        from pytorch_lightning.utilities.distributed import rank_zero_only
    except Exception:
        def rank_zero_only(fn):
            return fn


def tensor_to_uint8(x, value_range):
    x = x.detach().float().cpu()
    if value_range == "minus1_1":
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    x = x.permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def save_triplet(source, target, generated, path, title, info_lines=None):
    src = Image.fromarray(tensor_to_uint8(source, "0_1"))
    tgt = Image.fromarray(tensor_to_uint8(target, "minus1_1"))
    gen = Image.fromarray(tensor_to_uint8(generated, "minus1_1"))

    images = [src, tgt, gen]
    labels = ["Source", "Target", "Generated"]

    w, h = src.size
    pad = 10
    title_h = 28
    info_h = 20 * len(info_lines) if info_lines else 0
    header_h = 24

    canvas = Image.new(
        "RGB",
        (3 * w + 4 * pad, h + title_h + info_h + header_h + 3 * pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=(0, 0, 0))

    y_img = pad + header_h + title_h
    for i, (img, lab) in enumerate(zip(images, labels)):
        x0 = pad + i * (w + pad)
        draw.text((x0, pad + header_h), lab, fill=(0, 0, 0))
        canvas.paste(img, (x0, y_img))

    if info_lines:
        y0 = y_img + h + 5
        for j, line in enumerate(info_lines):
            draw.text((pad, y0 + 20 * j), line, fill=(0, 0, 0))

    canvas.save(path)


class FixedPoseVisualCallback(Callback):
    def __init__(
        self,
        data_root,
        step1_json_path,
        save_dir,
        tag="truepose",
        every_n_steps=1000,
        max_samples=64,
        batch_size=4,
        ddim_steps=20,
        cfg_scale=1.0,
        size_mult=1,
        image_size=128,
        num_workers=2,
        seed=2026,
    ):
        super().__init__()
        self.data_root = data_root
        self.step1_json_path = step1_json_path
        self.save_dir = save_dir
        self.tag = str(tag)
        self.every_n_steps = int(every_n_steps)
        self.max_samples = int(max_samples)
        self.batch_size = int(batch_size)
        self.ddim_steps = int(ddim_steps)
        self.cfg_scale = float(cfg_scale)
        self.size_mult = int(size_mult)
        self.image_size = int(image_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.dataset = None
        self.last_run_step = -1

    def setup(self, trainer, pl_module, stage=None):
        if self.dataset is None:
            print(f"[FixedVisual:{self.tag}] loading {self.step1_json_path}", flush=True)
            self.dataset = TestDatasetWithPrediction(
                self.step1_json_path,
                self.data_root,
                size_mult=self.size_mult,
                max_samples=self.max_samples,
                image_size=self.image_size,
            )
            print(f"[FixedVisual:{self.tag}] N={len(self.dataset)}", flush=True)

    @torch.no_grad()
    def _sample(self, pl_module, hint, heading, range_num):
        device = pl_module.device
        bs = hint.shape[0]

        cond_emb = pl_module.numeric_encoder(
            heading.to(device).float(),
            range_num.to(device).float(),
        )

        # Optional compatibility with later source-token variants.
        if getattr(pl_module, "enable_source_latent_tokens", False) and hasattr(pl_module, "encode_source_tokens_from_hint"):
            src_tokens = pl_module.encode_source_tokens_from_hint(hint.to(device).float())
            if src_tokens is not None:
                cond_emb = torch.cat([cond_emb, src_tokens], dim=1)

        cond = {
            "c_concat": [hint.to(device).float()],
            "c_crossattn": [cond_emb],
        }
        uc_cross = pl_module.get_unconditional_conditioning(bs)
        uc = {
            "c_concat": [hint.to(device).float()],
            "c_crossattn": [uc_cross],
        }

        shape = (
            pl_module.channels,
            hint.shape[2] // 8,
            hint.shape[3] // 8,
        )

        sampler = DDIMSampler(pl_module)
        samples, _ = sampler.sample(
            S=self.ddim_steps,
            batch_size=bs,
            shape=shape,
            conditioning=cond,
            verbose=False,
            unconditional_guidance_scale=self.cfg_scale,
            unconditional_conditioning=uc,
            eta=0.0,
        )
        return pl_module.decode_first_stage(samples)

    @rank_zero_only
    @torch.no_grad()
    def _run_visual_test(self, trainer, pl_module):
        if self.dataset is None or len(self.dataset) == 0:
            return

        step = int(trainer.global_step)
        out_dir = os.path.join(self.save_dir, self.tag, f"step_{step:06d}")
        boards_dir = os.path.join(out_dir, "boards")
        raw_dir = os.path.join(out_dir, "raw")
        os.makedirs(boards_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)

        print(f"[FixedVisual:{self.tag}] step={step} saving to {out_dir}", flush=True)

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

        was_training = pl_module.training
        pl_module.eval()

        summary = []
        global_idx = 0

        torch.manual_seed(self.seed + step)

        for batch in tqdm(loader, desc=f"FixedVisual {self.tag} step {step}", dynamic_ncols=True):
            hint = batch["hint"].permute(0, 3, 1, 2).contiguous().float().to(pl_module.device)
            target = batch["jpg"].permute(0, 3, 1, 2).contiguous().float().to(pl_module.device)
            pred_h = batch["pred_heading"].float().to(pl_module.device)
            pred_r = batch["pred_range"].float().to(pl_module.device)
            true_h = batch["true_heading"].float().to(pl_module.device)
            true_r = batch["true_range"].float().to(pl_module.device)

            generated = self._sample(pl_module, hint, pred_h, pred_r)

            bs = hint.shape[0]
            for i in range(bs):
                sid = f"{global_idx:04d}"
                one_raw = os.path.join(raw_dir, sid)
                os.makedirs(one_raw, exist_ok=True)

                Image.fromarray(tensor_to_uint8(hint[i], "0_1")).save(os.path.join(one_raw, "source.png"))
                Image.fromarray(tensor_to_uint8(target[i], "minus1_1")).save(os.path.join(one_raw, "target.png"))
                Image.fromarray(tensor_to_uint8(generated[i], "minus1_1")).save(os.path.join(one_raw, "generated.png"))

                board_path = os.path.join(boards_dir, f"{sid}.png")
                save_triplet(
                    hint[i],
                    target[i],
                    generated[i],
                    board_path,
                    title=f"{self.tag} | step={step}",
                    info_lines=[
                        f"pred_h={float(pred_h[i]):.2f}, pred_r={float(pred_r[i]):.2f}",
                        f"true_h={float(true_h[i]):.2f}, true_r={float(true_r[i]):.2f}",
                    ],
                )

                summary.append({
                    "idx": global_idx,
                    "board_path": board_path,
                    "pred_heading": float(pred_h[i]),
                    "pred_range": float(pred_r[i]),
                    "true_heading": float(true_h[i]),
                    "true_range": float(true_r[i]),
                })
                global_idx += 1

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if was_training:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self.every_n_steps <= 0:
            return
        if step <= 0:
            return
        if step == self.last_run_step:
            return
        if step % self.every_n_steps != 0:
            return
        self.last_run_step = step
        self._run_visual_test(trainer, pl_module)
