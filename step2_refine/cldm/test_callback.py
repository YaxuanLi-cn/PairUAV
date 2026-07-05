import os
import json
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import Dataset
from tqdm import tqdm
from ldm.models.diffusion.ddim import DDIMSampler

BASE_SIZE = 64  # Base unit size (must be divisible by 64 for UNet compatibility)


def center_crop_tensor(tensor, target_size):
    """Center crop a tensor to target_size. Tensor shape: B, C, H, W."""
    _, _, h, w = tensor.shape
    start_h = max((h - target_size) // 2, 0)
    start_w = max((w - target_size) // 2, 0)
    return tensor[:, :, start_h:start_h + target_size, start_w:start_w + target_size]


def angular_difference(angle1, angle2):
    """Minimum angular difference in degrees, handling wraparound."""
    diff = abs(float(angle1) - float(angle2))
    return min(diff, 360.0 - diff)


def _to_uint8_img(tensor, value_range='minus1_1'):
    """Convert a CHW tensor to uint8 RGB for saving."""
    x = tensor.detach().float().cpu()
    if value_range == 'minus1_1':
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def _save_labeled_grid(images, labels, path, value_ranges=None, ncols=3, cell_pad=6, label_h=34):
    """Save a small labeled image grid using PIL.

    images: list of CHW tensors.
    labels: list of short strings.
    value_ranges: list with '0_1' or 'minus1_1'.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if value_ranges is None:
        value_ranges = ['minus1_1'] * len(images)
    pil_imgs = []
    for img, vr in zip(images, value_ranges):
        pil_imgs.append(Image.fromarray(_to_uint8_img(img, value_range=vr)))
    if not pil_imgs:
        return
    w, h = pil_imgs[0].size
    n = len(pil_imgs)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    canvas_w = ncols * w + (ncols + 1) * cell_pad
    canvas_h = nrows * (h + label_h) + (nrows + 1) * cell_pad
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(pil_imgs, labels)):
        r = i // ncols
        c = i % ncols
        x0 = cell_pad + c * (w + cell_pad)
        y0 = cell_pad + r * (h + label_h + cell_pad)
        canvas.paste(im, (x0, y0 + label_h))
        draw.text((x0, y0 + 2), str(lab)[:64], fill=(0, 0, 0))
    canvas.save(path)


class TestDatasetWithPrediction(Dataset):
    """Test dataset that loads predictions from step1_seen.json.

    v4: max_samples is applied while building metadata, so quick eval really loads
    only a small subset instead of all 2.7M samples.
    """

    def __init__(self, step1_json_path, data_root='/root/autodl-tmp/dreamnav/try_test/', size_mult=1, max_samples=None, image_size=None):
        self.data = []
        self.data_root = data_root.rstrip('/')
        if image_size is None or int(image_size) <= 0:
            self.image_size = BASE_SIZE * int(size_mult)
        else:
            self.image_size = int(image_size)
        if self.image_size % 64 != 0:
            raise ValueError(f"image_size must be divisible by 64 for SD/ControlNet, got {self.image_size}")
        print(f"[Stage] Test image size: {self.image_size}x{self.image_size} (latent {self.image_size // 8}x{self.image_size // 8})", flush=True)

        print(f"[Stage] Loading Step1 prediction JSON: {step1_json_path}", flush=True)
        with open(step1_json_path, 'r') as f:
            step1_data = json.load(f)

        pred_deg = step1_data['pred_deg_num']
        true_deg = step1_data['true_deg_num']
        pred_rag = step1_data['pred_rag_num']
        true_rag = step1_data['true_rag_num']
        json_paths = step1_data['json_path']

        original_n = len(json_paths)
        if max_samples is not None:
            keep_n = min(max_samples, original_n)
            print(f"[Stage] Test metadata limit: {keep_n}/{original_n} samples", flush=True)
        else:
            keep_n = original_n
            print(f"[Stage] Test metadata limit: all {original_n} samples", flush=True)

        pred_deg = pred_deg[:keep_n]
        true_deg = true_deg[:keep_n]
        pred_rag = pred_rag[:keep_n]
        true_rag = true_rag[:keep_n]
        json_paths = json_paths[:keep_n]

        for i, json_path in enumerate(tqdm(json_paths, desc="Loading test json metadata", dynamic_ncols=True)):
            with open(json_path, 'r', encoding='utf-8') as f:
                item_json = json.load(f)

            a_path = self.data_root + '/tours/' + item_json["image_a"]
            b_path = self.data_root + '/tours/' + item_json["image_b"]

            self.data.append({
                'image_a': a_path,
                'image_b': b_path,
                'pred_heading': pred_deg[i],
                'true_heading': true_deg[i],
                'pred_range': pred_rag[i],
                'true_range': true_rag[i],
                'json_path': json_path,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source = cv2.imread(item['image_a'])
        target = cv2.imread(item['image_b'])
        if source is None:
            raise FileNotFoundError(item['image_a'])
        if target is None:
            raise FileNotFoundError(item['image_b'])

        source = cv2.resize(source, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return {
            'hint': source,
            'jpg': target,
            'pred_heading': item['pred_heading'],
            'true_heading': item['true_heading'],
            'pred_range': item['pred_range'],
            'true_range': item['true_range'],
        }


class EpochTestCallback(Callback):
    """Epoch-end evaluator and fixed-probe visualizer.

    It still supports the original DDIM image-MSE selector, but v4 also reports
    diagnostic metrics:
      - heading/range/joint candidate top-1 accuracy against oracle candidate
      - condition sensitivity: how different the 9 candidate generated images are
      - probe images: source/GT/generated and 3x3 candidate grids
    """

    def __init__(self,
                 data_root,
                 step1_json_path='/root/dreamNav/step1/step1_seen.json',
                 heading_offset=20.0,
                 range_offset=1.5,
                 ddim_steps=50,
                 batch_size=1,
                 max_test_samples=None,
                 save_dir=None,
                 test_batch_size=3,
                 size_mult=1,
                 image_size=None,
                 probe_save_dir=None,
                 num_probe_samples=8,
                 probe_ddim_steps=20,
                 enable_probe=True,
                 cfg_scale=9.0):
        super().__init__()
        self.data_root = data_root
        self.step1_json_path = step1_json_path
        self.heading_offset = heading_offset
        self.range_offset = range_offset
        self.ddim_steps = ddim_steps
        self.batch_size = batch_size
        self.max_test_samples = max_test_samples
        self.save_dir = save_dir
        self.test_dataset = None
        self.test_batch_size = test_batch_size
        self.num_offsets = 9
        self.size_mult = size_mult
        if image_size is None or int(image_size) <= 0:
            self.image_size = BASE_SIZE * int(size_mult)
        else:
            self.image_size = int(image_size)
        if self.image_size % 64 != 0:
            raise ValueError(f"image_size must be divisible by 64 for SD/ControlNet, got {self.image_size}")
        print(f"[Stage] EpochTestCallback image size: {self.image_size}x{self.image_size} (latent {self.image_size // 8}x{self.image_size // 8})", flush=True)
        self.probe_save_dir = probe_save_dir
        self.num_probe_samples = int(num_probe_samples)
        self.probe_ddim_steps = int(probe_ddim_steps)
        self.enable_probe = bool(enable_probe)
        self.cfg_scale = float(cfg_scale)

    def setup(self, trainer, pl_module, stage=None):
        if self.test_dataset is None:
            print(f"[Stage] Loading test dataset from {self.step1_json_path}...", flush=True)
            self.test_dataset = TestDatasetWithPrediction(
                self.step1_json_path,
                self.data_root,
                size_mult=self.size_mult,
                max_samples=self.max_test_samples,
                image_size=self.image_size,
            )
            print(f"[Stage] Test dataset loaded with {len(self.test_dataset)} samples", flush=True)

    def _offset_combos(self):
        heading_offsets = [-self.heading_offset, 0.0, self.heading_offset]
        range_offsets = [-self.range_offset, 0.0, self.range_offset]
        return heading_offsets, range_offsets, [(h, r) for h in heading_offsets for r in range_offsets]

    @torch.no_grad()
    def _sample_images(self, pl_module, hints_batch, heading_tensor, range_tensor, ddim_steps):
        device = pl_module.device
        total_bs = hints_batch.shape[0]
        cond_emb = pl_module.numeric_encoder(heading_tensor.to(device).float(), range_tensor.to(device).float())
        cond = {"c_concat": [hints_batch.to(device)], "c_crossattn": [cond_emb]}
        uc_cross = pl_module.get_unconditional_conditioning(total_bs)
        uc_full = {"c_concat": [hints_batch.to(device)], "c_crossattn": [uc_cross]}
        shape = (pl_module.channels, hints_batch.shape[2] // 8, hints_batch.shape[3] // 8)
        ddim_sampler = DDIMSampler(pl_module)
        samples, _ = ddim_sampler.sample(
            ddim_steps, total_bs, shape, cond,
            verbose=False,
            unconditional_guidance_scale=self.cfg_scale,
            unconditional_conditioning=uc_full,
        )
        return pl_module.decode_first_stage(samples)

    @rank_zero_only
    @torch.no_grad()
    def _save_probe_visualizations(self, trainer, pl_module):
        if not self.enable_probe or self.probe_save_dir is None or self.num_probe_samples <= 0:
            return
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return

        epoch = trainer.current_epoch
        out_dir = os.path.join(self.probe_save_dir, f"epoch_{epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[Stage] Saving fixed probe visualizations to {out_dir}", flush=True)

        _, _, offset_combos = self._offset_combos()
        n = min(self.num_probe_samples, len(self.test_dataset))
        summary = []

        for probe_i in tqdm(range(n), desc=f"Epoch {epoch} fixed probes", dynamic_ncols=True):
            sample = self.test_dataset[probe_i]
            hint = torch.from_numpy(sample['hint']).permute(2, 0, 1).contiguous().float().to(pl_module.device)
            target = torch.from_numpy(sample['jpg']).permute(2, 0, 1).contiguous().float().to(pl_module.device)

            # A. True-condition generation: source / target / generated(true heading/range)
            true_h = torch.tensor([sample['true_heading']], device=pl_module.device).float()
            true_r = torch.tensor([sample['true_range']], device=pl_module.device).float()
            gen_true = self._sample_images(
                pl_module,
                hint.unsqueeze(0),
                true_h,
                true_r,
                self.probe_ddim_steps,
            )[0]
            triplet_path = os.path.join(out_dir, f"probe_{probe_i:03d}_triplet.png")
            _save_labeled_grid(
                [hint, target, gen_true],
                ["Source", "GT Target", f"Gen true h={float(true_h[0]):.1f} r={float(true_r[0]):.1f}"],
                triplet_path,
                value_ranges=['0_1', 'minus1_1', 'minus1_1'],
                ncols=3,
            )

            # B/C. 9-candidate grid around Stage1 prediction.
            all_hints = []
            all_headings = []
            all_ranges = []
            labels = []
            pred_h = float(sample['pred_heading'])
            true_h_val = float(sample['true_heading'])
            pred_r = float(sample['pred_range'])
            true_r_val = float(sample['true_range'])
            for h_off, r_off in offset_combos:
                all_hints.append(hint)
                cand_h = pred_h + h_off
                cand_r = pred_r + r_off
                all_headings.append(cand_h)
                all_ranges.append(cand_r)
                labels.append(f"dh={h_off:+.0f} dr={r_off:+.1f}")

            hints_batch = torch.stack(all_hints).to(pl_module.device)
            heading_tensor = torch.tensor(all_headings, device=pl_module.device).float()
            range_tensor = torch.tensor(all_ranges, device=pl_module.device).float()
            generated = self._sample_images(pl_module, hints_batch, heading_tensor, range_tensor, self.probe_ddim_steps)
            generated_cropped = center_crop_tensor(generated, self.image_size)
            target_repeat = target.unsqueeze(0).repeat(self.num_offsets, 1, 1, 1)
            scores = torch.mean((generated_cropped - target_repeat) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
            best_idx = int(np.argmin(scores))
            center = generated_cropped[4:5]
            sensitivity = torch.mean((generated_cropped - center) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
            sensitivity_mean = float(np.mean(np.delete(sensitivity, 4)))

            grid_labels = []
            for j, lab in enumerate(labels):
                mark = "*" if j == best_idx else " "
                grid_labels.append(f"{mark}{lab} MSE={scores[j]:.3f}")
            grid_path = os.path.join(out_dir, f"probe_{probe_i:03d}_candidate_grid.png")
            _save_labeled_grid(
                [generated_cropped[j] for j in range(self.num_offsets)],
                grid_labels,
                grid_path,
                value_ranges=['minus1_1'] * self.num_offsets,
                ncols=3,
                label_h=42,
            )

            best_h_off, best_r_off = offset_combos[best_idx]
            summary.append({
                'probe_idx': probe_i,
                'pred_heading': pred_h,
                'true_heading': true_h_val,
                'pred_range': pred_r,
                'true_range': true_r_val,
                'chosen_offset_heading': best_h_off,
                'chosen_offset_range': best_r_off,
                'chosen_score': float(scores[best_idx]),
                'condition_sensitivity_mse': sensitivity_mean,
                'triplet_path': triplet_path,
                'candidate_grid_path': grid_path,
            })

        with open(os.path.join(out_dir, 'probe_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[Stage] Probe visualizations saved: {n} samples", flush=True)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        ckpt_dir = os.path.join(trainer.default_root_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'epoch={trainer.current_epoch}.ckpt')
        save_every = int(os.environ.get("SAVE_EVERY_N_EPOCHS", "1"))
        should_save = ((trainer.current_epoch + 1) % save_every == 0) or ((trainer.current_epoch + 1) == trainer.max_epochs)
        if os.environ.get("DISABLE_EPOCH_CKPT", "0") != "1" and should_save:
            trainer.save_checkpoint(ckpt_path)
            print(f"\nCheckpoint saved to {ckpt_path}")
        else:
            print(f"\n[Skip] checkpoint save disabled/skipped: {ckpt_path}")

        pl_module.eval()
        self._save_probe_visualizations(trainer, pl_module)

        print(f"\n{'=' * 60}")
        print(f"Running epoch {trainer.current_epoch} testing with heading_offset={self.heading_offset}, range_offset={self.range_offset}")
        print(f"{'=' * 60}")

        test_indices = range(len(self.test_dataset))
        if self.max_test_samples is not None:
            test_indices = range(min(self.max_test_samples, len(self.test_dataset)))

        original_errors_heading = []
        corrected_errors_heading = []
        original_errors_range = []
        corrected_errors_range = []
        best_offsets_heading = []
        best_offsets_range = []
        selector_acc_heading = []
        selector_acc_range = []
        selector_acc_joint = []
        condition_sensitivity_scores = []

        device = pl_module.device
        heading_offsets, range_offsets, offset_combos = self._offset_combos()

        with torch.no_grad():
            for batch_start in tqdm(range(0, len(test_indices), self.test_batch_size), desc=f"Epoch {trainer.current_epoch} testing", dynamic_ncols=True):
                batch_idx_list = list(test_indices[batch_start:batch_start + self.test_batch_size])
                actual_bs = len(batch_idx_list)
                total_bs = actual_bs * self.num_offsets

                all_hints = []
                all_targets = []
                all_headings = []
                all_ranges = []
                batch_pred_headings = []
                batch_true_headings = []
                batch_pred_ranges = []
                batch_true_ranges = []

                for idx in batch_idx_list:
                    sample = self.test_dataset[idx]
                    hint = torch.from_numpy(sample['hint']).permute(2, 0, 1).contiguous().float()
                    target = torch.from_numpy(sample['jpg']).permute(2, 0, 1).contiguous().float()

                    pred_heading = float(sample['pred_heading'])
                    true_heading = float(sample['true_heading'])
                    pred_range = float(sample['pred_range'])
                    true_range = float(sample['true_range'])

                    batch_pred_headings.append(pred_heading)
                    batch_true_headings.append(true_heading)
                    batch_pred_ranges.append(pred_range)
                    batch_true_ranges.append(true_range)

                    for h_off, r_off in offset_combos:
                        all_hints.append(hint)
                        all_targets.append(target)
                        all_headings.append(pred_heading + h_off)
                        all_ranges.append(pred_range + r_off)

                hints_batch = torch.stack(all_hints).to(device)
                targets_batch = torch.stack(all_targets).to(device)
                heading_tensor = torch.tensor(all_headings, device=device).float()
                range_tensor = torch.tensor(all_ranges, device=device).float()

                generated = self._sample_images(pl_module, hints_batch, heading_tensor, range_tensor, self.ddim_steps)
                generated_cropped = center_crop_tensor(generated, self.image_size)
                targets_cropped = center_crop_tensor(targets_batch, self.image_size)

                mse_per_item = torch.mean((generated_cropped - targets_cropped) ** 2, dim=(1, 2, 3))
                mse_per_sample = mse_per_item.view(actual_bs, self.num_offsets)
                best_indices = mse_per_sample.argmin(dim=1)

                # Condition sensitivity: average generated image change relative to center candidate.
                gen_view = generated_cropped.view(actual_bs, self.num_offsets, *generated_cropped.shape[1:])
                center = gen_view[:, 4:5]
                sens = torch.mean((gen_view - center) ** 2, dim=(2, 3, 4))
                if self.num_offsets > 1:
                    sens_non_center = torch.cat([sens[:, :4], sens[:, 5:]], dim=1)
                    condition_sensitivity_scores.extend(sens_non_center.mean(dim=1).detach().cpu().tolist())

                for i in range(actual_bs):
                    best_idx = int(best_indices[i].item())
                    best_h_off, best_r_off = offset_combos[best_idx]
                    pred_heading = batch_pred_headings[i]
                    true_heading = batch_true_headings[i]
                    pred_range = batch_pred_ranges[i]
                    true_range = batch_true_ranges[i]

                    best_heading = pred_heading + best_h_off
                    best_range = pred_range + best_r_off

                    original_heading_error = angular_difference(pred_heading, true_heading)
                    corrected_heading_error = angular_difference(best_heading, true_heading)
                    original_range_error = abs(pred_range - true_range)
                    corrected_range_error = abs(best_range - true_range)

                    # Oracle candidate diagnostics.
                    h_errs = np.array([angular_difference(pred_heading + h, true_heading) for h, _ in offset_combos])
                    r_errs = np.array([abs(pred_range + r - true_range) for _, r in offset_combos])
                    joint_errs = h_errs / max(abs(self.heading_offset), 1e-6) + r_errs / max(abs(self.range_offset), 1e-6)
                    selector_acc_heading.append(float(np.isclose(h_errs[best_idx], h_errs.min())))
                    selector_acc_range.append(float(np.isclose(r_errs[best_idx], r_errs.min())))
                    selector_acc_joint.append(float(best_idx == int(joint_errs.argmin())))

                    original_errors_heading.append(original_heading_error)
                    corrected_errors_heading.append(corrected_heading_error)
                    original_errors_range.append(original_range_error)
                    corrected_errors_range.append(corrected_range_error)
                    best_offsets_heading.append(best_h_off)
                    best_offsets_range.append(best_r_off)

        original_mae_heading = float(np.mean(original_errors_heading))
        corrected_mae_heading = float(np.mean(corrected_errors_heading))
        original_mse_heading = float(np.mean(np.array(original_errors_heading) ** 2))
        corrected_mse_heading = float(np.mean(np.array(corrected_errors_heading) ** 2))
        original_mae_range = float(np.mean(original_errors_range))
        corrected_mae_range = float(np.mean(corrected_errors_range))
        original_mse_range = float(np.mean(np.array(original_errors_range) ** 2))
        corrected_mse_range = float(np.mean(np.array(corrected_errors_range) ** 2))
        offset_counts_heading = {o: best_offsets_heading.count(o) for o in heading_offsets}
        offset_counts_range = {o: best_offsets_range.count(o) for o in range_offsets}
        acc_heading = float(np.mean(selector_acc_heading)) if selector_acc_heading else 0.0
        acc_range = float(np.mean(selector_acc_range)) if selector_acc_range else 0.0
        acc_joint = float(np.mean(selector_acc_joint)) if selector_acc_joint else 0.0
        cond_sens = float(np.mean(condition_sensitivity_scores)) if condition_sensitivity_scores else 0.0

        print(f"\n{'=' * 60}")
        print(f"Epoch {trainer.current_epoch} Test Results (heading_offset={self.heading_offset}, range_offset={self.range_offset})")
        print(f"{'=' * 60}")
        print("\nHeading:")
        print(f"  Original MAE: {original_mae_heading:.4f}")
        print(f"  Corrected MAE: {corrected_mae_heading:.4f}")
        print(f"  Original MSE: {original_mse_heading:.4f}")
        print(f"  Corrected MSE: {corrected_mse_heading:.4f}")
        print(f"  Improvement MAE: {original_mae_heading - corrected_mae_heading:.4f}")
        print(f"  Offset usage: {offset_counts_heading}")
        print("\nRange:")
        print(f"  Original MAE: {original_mae_range:.4f}")
        print(f"  Corrected MAE: {corrected_mae_range:.4f}")
        print(f"  Original MSE: {original_mse_range:.4f}")
        print(f"  Corrected MSE: {corrected_mse_range:.4f}")
        print(f"  Improvement MAE: {original_mae_range - corrected_mae_range:.4f}")
        print(f"  Offset usage: {offset_counts_range}")
        print("\nSelector diagnostics:")
        print(f"  Candidate Top-1 Acc Heading: {acc_heading:.4f}")
        print(f"  Candidate Top-1 Acc Range:   {acc_range:.4f}")
        print(f"  Candidate Top-1 Acc Joint:   {acc_joint:.4f}")
        print(f"  Condition sensitivity MSE:   {cond_sens:.6f}")
        print(f"{'=' * 60}\n")

        pl_module.log('test/original_mae_heading', original_mae_heading, on_epoch=True)
        pl_module.log('test/corrected_mae_heading', corrected_mae_heading, on_epoch=True)
        pl_module.log('test/original_mae_range', original_mae_range, on_epoch=True)
        pl_module.log('test/corrected_mae_range', corrected_mae_range, on_epoch=True)
        pl_module.log('test/selector_acc_heading', acc_heading, on_epoch=True)
        pl_module.log('test/selector_acc_range', acc_range, on_epoch=True)
        pl_module.log('test/selector_acc_joint', acc_joint, on_epoch=True)
        pl_module.log('test/condition_sensitivity_mse', cond_sens, on_epoch=True)

        if self.save_dir:
            results = {
                'epoch': trainer.current_epoch,
                'heading_offset': self.heading_offset,
                'range_offset': self.range_offset,
                'ddim_steps': self.ddim_steps,
                'max_test_samples': self.max_test_samples,
                'test_batch_size': self.test_batch_size,
                'heading': {
                    'original_mae': original_mae_heading,
                    'corrected_mae': corrected_mae_heading,
                    'improvement_mae': original_mae_heading - corrected_mae_heading,
                    'original_mse': original_mse_heading,
                    'corrected_mse': corrected_mse_heading,
                    'offset_counts': offset_counts_heading,
                },
                'range': {
                    'original_mae': original_mae_range,
                    'corrected_mae': corrected_mae_range,
                    'improvement_mae': original_mae_range - corrected_mae_range,
                    'original_mse': original_mse_range,
                    'corrected_mse': corrected_mse_range,
                    'offset_counts': offset_counts_range,
                },
                'selector_diagnostics': {
                    'candidate_top1_acc_heading': acc_heading,
                    'candidate_top1_acc_range': acc_range,
                    'candidate_top1_acc_joint': acc_joint,
                    'condition_sensitivity_mse': cond_sens,
                },
            }
            os.makedirs(self.save_dir, exist_ok=True)
            result_path = os.path.join(self.save_dir, f'test_results_epoch_{trainer.current_epoch}.json')
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {result_path}")

        pl_module.train()
