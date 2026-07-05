import argparse
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.lora_utils import inject_lora_into_linear, mark_only_lora_trainable
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.test_callback import EpochTestCallback
from cldm.fixed_visual_callback import FixedPoseVisualCallback
from cldm.lr_warmup_callback import LinearWarmupLRCallback
from cldm.rgb_condition_supervision import patch_model_with_rgb_condition_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Step2 Training')
    # ==================== Configs ====================
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the pretrained checkpoint (.ckpt)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=-1, help='Stop training after this many optimizer steps. Use -1 to disable.')
    parser.add_argument('--logger_freq', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=0, help='Linear learning-rate warmup steps')
    parser.add_argument('--warmup_min_lr_scale', type=float, default=0.0, help='Initial LR scale during warmup')

    # ==================== Frozen RGB condition loss ====================
    parser.add_argument('--rgb_loss_checkpoint', type=str, default=None,
                        help='Path to frozen RGB condition predictor best.pt. If None, RGB loss is disabled.')
    parser.add_argument('--rgb_loss_weight', type=float, default=0.0,
                        help='Weight for frozen RGB condition loss')
    parser.add_argument('--rgb_loss_start_step', type=int, default=0,
                        help='Start applying RGB condition loss after this global step')
    parser.add_argument('--rgb_loss_every_n_steps', type=int, default=1,
                        help='Apply RGB loss every N training steps')
    parser.add_argument('--rgb_loss_batch_size', type=int, default=8,
                        help='Subset size for RGB loss to control overhead')
    parser.add_argument('--rgb_loss_model', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument('--rgb_loss_image_size', type=int, default=128,
                        help='Input resolution expected by RGB predictor')
    parser.add_argument('--rgb_loss_range_scale', type=float, default=100.0,
                        help='Range scaling used when training RGB predictor')
    parser.add_argument('--rgb_loss_angle_weight', type=float, default=1.0)
    parser.add_argument('--rgb_loss_range_weight', type=float, default=1.0)
    parser.add_argument('--rgb_loss_norm_weight', type=float, default=0.01)
    parser.add_argument('--rgb_loss_t_min', type=int, default=0,
                        help='Min diffusion timestep used for auxiliary x0 prediction')
    parser.add_argument('--rgb_loss_t_max', type=int, default=500,
                        help='Max diffusion timestep used for auxiliary x0 prediction; <=0 means full range')
    parser.add_argument('--rgb_loss_print_every', type=int, default=500)
    parser.add_argument('--rgb_loss_plot_every', type=int, default=1000,
                        help='Save RGB/base loss curve png every N global steps. Set <=0 to disable.')
    parser.add_argument('--rgb_loss_plot_dir', type=str, default=None,
                        help='Directory to save RGB loss curves; defaults to output_dir/rgb_loss_curves')
    # NOTE: the old code used action='store_true', default=True, which makes sd_locked
    # impossible to turn off from CLI. Keep it for compatibility, but use train_mode below
    # to explicitly decide what is optimized.
    parser.add_argument('--sd_locked', action='store_true', default=True)
    parser.add_argument('--only_mid_control', action='store_true', default=False)

    # ==================== Model Config ====================
    parser.add_argument('--config_path', type=str, default='./cldm_v15_numeric.yaml',
                        help='Model YAML config. Use cldm_v15_pose_*.yaml for pose-token experiments.')

    # ==================== Trainable Scope Configs ====================
    parser.add_argument('--train_mode', type=str, default='control',
                        choices=['control', 'decoder', 'control_decoder',
                                 'lora_decoder', 'lora_control_decoder',
                                 'lora_control_decoder_hint'],
                        help=("control: original behavior, train ControlNet + numeric encoder; "
                              "decoder: freeze ControlNet and tune UNet decoder/output blocks + numeric encoder; "
                              "control_decoder: train both ControlNet and UNet decoder/output blocks; "
                              "lora_decoder: freeze base UNet decoder and train LoRA adapters in decoder linear layers + numeric encoder; "
                              "lora_control_decoder: LoRA adapters in ControlNet and UNet decoder + numeric encoder; "
                              "lora_control_decoder_hint: same as lora_control_decoder, plus train ControlNet hint conv gates/zero-convs so source image can actually affect UNet."))
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=8.0)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=32)

    # ==================== Dataset Configs ====================
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory of the dataset (e.g. /root/autodl-tmp/dreamnav/)')
    parser.add_argument('--dataset_type', type=str, default='try_train',
                        help='Dataset subdirectory name (e.g. try_train)')

    # ==================== Test Callback Configs ====================
    # Path to step1 prediction JSON file
    parser.add_argument('--step1_json_path', type=str, required=True,
                        help='Path to step1 prediction JSON file')
    # Correction offsets: will try prediction ± these values and 0
    parser.add_argument('--heading_offset', type=float, default=10.0,
                        help='heading will try ±this value')
    parser.add_argument('--range_offset', type=float, default=1.5,
                        help='range will try ±this value')
    # DDIM sampling steps for test image generation
    parser.add_argument('--test_ddim_steps', type=int, default=50)
    # Maximum test samples per epoch (None for all, set smaller for faster testing)
    parser.add_argument('--max_test_samples', type=int, default=100,
                        help='Set to -1 to test all samples')
    # Number of test samples to process simultaneously (each has 9 offset combos)
    parser.add_argument('--test_batch_size', type=int, default=3,
                        help='Number of test samples to run in parallel (total batch = this * 9)')
    # Directory to save test results
    parser.add_argument('--test_save_dir', type=str, required=True,
                        help='Directory to save test results')

    # ==================== Probe Visualization Configs ====================
    parser.add_argument('--probe_save_dir', type=str, default=None,
                        help='Directory to save fixed probe visualizations; defaults to output_dir/probes')
    parser.add_argument('--num_probe_samples', type=int, default=8,
                        help='Number of fixed test samples used for source/target/generated and 3x3 candidate grids')
    parser.add_argument('--probe_ddim_steps', type=int, default=20,
                        help='DDIM steps for fixed probe visualizations')
    parser.add_argument('--disable_probe', action='store_true',
                        help='Disable fixed probe visualization')
    parser.add_argument('--cfg_scale', type=float, default=9.0,
                        help='Classifier-free guidance scale used in test/probe sampling')


    # ==================== Fixed Visual Monitoring Configs ====================
    parser.add_argument('--fixed_true_json_path', type=str, default=None,
                        help='Fixed test JSON for true-pose visual monitoring')
    parser.add_argument('--fixed_pred_json_path', type=str, default=None,
                        help='Fixed test JSON for predicted-pose visual monitoring')
    parser.add_argument('--fixed_visual_save_dir', type=str, default=None,
                        help='Directory for fixed visual monitoring outputs; defaults to output_dir/fixed_visuals')
    parser.add_argument('--fixed_visual_every_n_steps', type=int, default=1000,
                        help='Run fixed visual monitoring every N optimizer steps')
    parser.add_argument('--fixed_visual_max_samples', type=int, default=64,
                        help='Number of fixed samples to render every time')
    parser.add_argument('--fixed_visual_batch_size', type=int, default=4,
                        help='Batch size for fixed visual monitoring')
    parser.add_argument('--fixed_visual_ddim_steps', type=int, default=20,
                        help='DDIM steps for fixed visual monitoring')
    parser.add_argument('--fixed_visual_num_workers', type=int, default=2,
                        help='DataLoader workers for fixed visual monitoring')

    # Output root directory for trainer
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory for training outputs (checkpoints, logs)')

    # ==================== Image Size Config ====================
    parser.add_argument('--size_mult', type=int, default=1,
                        help='Backward-compatible image size multiplier (1=64x64, 2=128x128, 4=256x256). Ignored when --image_size > 0.')
    parser.add_argument('--image_size', type=int, default=0,
                        help='Direct image resolution. Use 128 or 256. Must be divisible by 64. If >0, overrides --size_mult.')

    return parser.parse_args()



def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def configure_trainable_scope(model, args):
    """Configure which parts of Step2 are updated.

    This directly addresses the mentor's question: what is fixed and what is fine-tuned?
    VAE is always frozen. The train_mode controls whether we train the original
    ControlNet branch, the SD/UNet decoder/output blocks, or LoRA adapters there.
    """
    # Start from a fully frozen model, then explicitly unfreeze selected modules.
    for p in model.parameters():
        p.requires_grad = False

    # VAE / first stage remains fixed by design.
    if hasattr(model, 'first_stage_model'):
        set_requires_grad(model.first_stage_model, False)

    # Numeric condition encoder is always trainable; otherwise heading/range cannot adapt.
    set_requires_grad(model.numeric_encoder, True)

    # Original behavior: train ControlNet/hint branch + numeric encoder.
    if args.train_mode in ['control', 'control_decoder']:
        set_requires_grad(model.control_model, True)

    # Mentor-style simple baseline: tune decoder/output side of SD/UNet.
    if args.train_mode in ['decoder', 'control_decoder']:
        set_requires_grad(model.model.diffusion_model.output_blocks, True)
        set_requires_grad(model.model.diffusion_model.out, True)

    # LoRA variants: freeze base weights, inject small adapters in selected Linear layers.
    if args.train_mode in ['lora_decoder', 'lora_control_decoder', 'lora_control_decoder_hint']:
        wrapped_decoder = inject_lora_into_linear(
            model.model.diffusion_model.output_blocks,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        wrapped_control = 0
        if args.train_mode in ['lora_control_decoder', 'lora_control_decoder_hint']:
            wrapped_control = inject_lora_into_linear(
                model.control_model,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
        # Freeze everything in selected modules except LoRA adapter weights.
        mark_only_lora_trainable(model.model.diffusion_model.output_blocks)
        if args.train_mode in ['lora_control_decoder', 'lora_control_decoder_hint']:
            mark_only_lora_trainable(model.control_model)
        else:
            set_requires_grad(model.control_model, False)
        print(f"[LoRA] wrapped decoder Linear layers: {wrapped_decoder}")
        print(f"[LoRA] wrapped control Linear layers: {wrapped_control}")

        if args.train_mode == 'lora_control_decoder_hint':
            # IMPORTANT: the source image enters ControlNet through convolutional hint
            # blocks and zero-conv gates. Linear LoRA alone does not train these convs.
            # control_sd15_ini.ckpt initializes these gates to zero, so if they stay
            # frozen, the source image can be passed in but still have almost no effect.
            set_requires_grad(model.control_model.input_hint_block, True)
            set_requires_grad(model.control_model.zero_convs, True)
            set_requires_grad(model.control_model.middle_block_out, True)
            print("[Hint] Enabled trainable ControlNet input_hint_block + zero_convs + middle_block_out")

    # Store for configure_optimizers() diagnostics.
    model.train_mode = args.train_mode

    total, trainable = count_params(model)
    print("\n========== Step2 trainable scope ==========")
    print(f"train_mode: {args.train_mode}")
    print(f"total params:     {total:,}")
    print(f"trainable params: {trainable:,} ({trainable / max(total, 1):.6%})")
    module_list = [
        ('VAE / first_stage_model', getattr(model, 'first_stage_model', None)),
        (f'numeric_encoder ({model.numeric_encoder.__class__.__name__})', getattr(model, 'numeric_encoder', None)),
        ('control_model', getattr(model, 'control_model', None)),
        ('Control input_hint_block', getattr(model.control_model, 'input_hint_block', None)),
        ('Control zero_convs', getattr(model.control_model, 'zero_convs', None)),
        ('Control middle_block_out', getattr(model.control_model, 'middle_block_out', None)),
        ('UNet input_blocks', model.model.diffusion_model.input_blocks),
        ('UNet middle_block', model.model.diffusion_model.middle_block),
        ('UNet output_blocks', model.model.diffusion_model.output_blocks),
        ('UNet out', model.model.diffusion_model.out),
    ]
    for name, module in module_list:
        if module is None:
            continue
        t, tr = count_params(module)
        print(f"{name:26s}: trainable {tr:>14,} / {t:>14,}")
    print("===========================================\n")

def main():
    args = parse_args()

    max_test_samples = None if args.max_test_samples < 0 else args.max_test_samples

    # Resolve image resolution. Prefer --image_size for clarity; keep --size_mult for old scripts.
    effective_image_size = int(args.image_size) if int(args.image_size) > 0 else int(args.size_mult) * 64
    if effective_image_size not in (64, 128, 192, 256, 320, 384, 512):
        print(f"[Warn] Unusual image_size={effective_image_size}. 128/256 are recommended.", flush=True)
    if effective_image_size % 64 != 0:
        raise ValueError(f"image_size must be divisible by 64, got {effective_image_size}")
    print(f"[Stage] Effective image resolution: {effective_image_size}x{effective_image_size}; latent {effective_image_size // 8}x{effective_image_size // 8}", flush=True)

    print(f"[Stage] Creating model from {args.config_path}", flush=True)
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config_path).cpu()
    print(f"[Stage] Loading checkpoint: {args.checkpoint_path}", flush=True)
    model.load_state_dict(load_state_dict(args.checkpoint_path, location='cpu'), strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    print(f"[Stage] Configuring trainable scope: {args.train_mode}", flush=True)
    configure_trainable_scope(model, args)

    # Optional frozen RGB condition predictor loss. This patches training_step only;
    # it does not change model architecture or trainable scope.
    patch_model_with_rgb_condition_loss(model, args, effective_image_size)

    if args.probe_save_dir is None:
        args.probe_save_dir = os.path.join(args.output_dir, 'probes')

    # Misc
    dataset = MyDataset(root_dir=args.root_dir, dataset_type=args.dataset_type, size_mult=args.size_mult, image_size=effective_image_size)
    print(f"[Stage] Creating DataLoader: batch_size={args.batch_size}, num_workers={args.num_workers}", flush=True)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    # 每12200保存+测试一次
    # Callbacks
    logger = ImageLogger(
        batch_frequency=args.logger_freq,
        log_images_kwargs={
            "sample": True,
            "ddim_steps": args.probe_ddim_steps,
            "unconditional_guidance_scale": args.cfg_scale,
        },
    )
    test_callback = EpochTestCallback(
        data_root=args.root_dir,
        step1_json_path=args.step1_json_path,
        heading_offset=args.heading_offset,
        range_offset=args.range_offset,
        ddim_steps=args.test_ddim_steps,
        max_test_samples=max_test_samples,
        save_dir=args.test_save_dir,
        test_batch_size=args.test_batch_size,
        size_mult=args.size_mult,
        image_size=effective_image_size,
        probe_save_dir=args.probe_save_dir,
        num_probe_samples=args.num_probe_samples,
        probe_ddim_steps=args.probe_ddim_steps,
        enable_probe=not args.disable_probe,
        cfg_scale=args.cfg_scale,
    )

    fixed_callbacks = []
    fixed_visual_save_dir = args.fixed_visual_save_dir or os.path.join(args.output_dir, 'fixed_visuals')

    if args.fixed_true_json_path:
        fixed_callbacks.append(FixedPoseVisualCallback(
            data_root=args.root_dir,
            step1_json_path=args.fixed_true_json_path,
            save_dir=fixed_visual_save_dir,
            tag='truepose',
            every_n_steps=args.fixed_visual_every_n_steps,
            max_samples=args.fixed_visual_max_samples,
            batch_size=args.fixed_visual_batch_size,
            ddim_steps=args.fixed_visual_ddim_steps,
            cfg_scale=args.cfg_scale,
            size_mult=args.size_mult,
            image_size=effective_image_size,
            num_workers=args.fixed_visual_num_workers,
        ))

    if args.fixed_pred_json_path:
        fixed_callbacks.append(FixedPoseVisualCallback(
            data_root=args.root_dir,
            step1_json_path=args.fixed_pred_json_path,
            save_dir=fixed_visual_save_dir,
            tag='predictpose',
            every_n_steps=args.fixed_visual_every_n_steps,
            max_samples=args.fixed_visual_max_samples,
            batch_size=args.fixed_visual_batch_size,
            ddim_steps=args.fixed_visual_ddim_steps,
            cfg_scale=args.cfg_scale,
            size_mult=args.size_mult,
            image_size=effective_image_size,
            num_workers=args.fixed_visual_num_workers,
        ))

    callbacks = [logger, test_callback] + fixed_callbacks
    if args.warmup_steps and args.warmup_steps > 0:
        callbacks.append(LinearWarmupLRCallback(
            base_lr=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_lr_scale=args.warmup_min_lr_scale,
            verbose=True,
        ))

    trainer = pl.Trainer(
        gpus=args.gpus, 
        precision=args.precision, 
        max_epochs=args.max_epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        callbacks=callbacks, 
        default_root_dir=args.output_dir,
        #limit_train_batches=1,  # 取消注释可用于快速测试
    )


    # Train!
    print("[Stage] Starting PyTorch Lightning training", flush=True)
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
