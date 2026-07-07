<h1 align="center">Last-Meter Precision Navigation for UAVs: A Diffusion-Refined Aerial Visual Servoing Approach</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2607.04352">
    <img src="https://img.shields.io/badge/arXiv-2607.04352-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/YaxuanLi/UAVM_2026_test">
    <img src="https://img.shields.io/badge/Dataset-HF%20Data-d8b04c?style=for-the-badge" alt="Dataset">
  </a>
  <a href="https://www.zdzheng.xyz/ACMMM2026Workshop-UAV/">
    <img src="https://img.shields.io/badge/Workshop-Page-d46a5a?style=for-the-badge" alt="Workshop">
  </a>
  <a href="mailto:yaxuanli.cn@gmail.com">
    <img src="https://img.shields.io/badge/Email-YaxuanLi-6b6b6b?style=for-the-badge" alt="Email">
  </a>
</p>

<hr />
## Project Structure

```text
UAVM_2026/
├── models/
│   ├── dino_resnet/
│   └── controlnet/
├── pairUAV/
│   ├── data_process.sh
│   └── University-Release.zip
├── baseline/
│   ├── SuperGlue/
│   ├── train.py
│   └── run.sh
└── step2_refine/
    ├── train_rgb_loss.py
    ├── train_rgb_condition_predictor.py
    ├── tutorial_dataset.py
    ├── cldm/
    ├── ldm/
    ├── cldm_v15_pose_hybrid.yaml
    ├── train_step2_example.sh
    └── train_rgb_condition_predictor_example.sh
```

---

## 1. Environment Setup

Create a unified conda environment for the baseline:

```bash
conda create -n uavm python=3.9
conda activate uavm

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

huggingface-cli download Ramos-Ramos/dino-resnet-50 --local-dir models/dino_resnet
```

For Stage-II diffusion training, additional dependencies from latent diffusion / ControlNet may be required. Please install the dependencies listed in the Step-II environment file if provided.

---

## 2. Data Preparation

### 2.1 Download University-1652 Dataset

Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request. You may use the [request template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

### 2.2 Download and Process PairUAV Dataset

Download and process the PairUAV dataset:

```bash
cd pairUAV/
bash data_process.sh
cd ..
```

This script downloads the dataset from HuggingFace and extracts train/test/tours data to the `pairUAV/` directory.

---

## 3. Stage-I: SuperGlue-Based Coarse Pose Estimation

### 3.1 Run SuperGlue Feature Matching

First, perform feature matching on image pairs:

```bash
cd baseline/SuperGlue

# Option 1: download precomputed matching results.
bash download_results.sh
cd ..

# Option 2: run feature matching.
python gen_test_pairs.py
bash run_train.sh
bash run_test.sh
cd ..
```

This generates matching results in `train_matches_data/` and `test_matches_data/`.

### 3.2 Train Stage-I Model

```bash
cd baseline/
bash run.sh
cd ..
```

The Stage-I model predicts coarse heading and range from an image pair. The predicted pose can be exported as a JSON file and used as the condition input for Stage-II diffusion refinement.

### 3.3 Evaluate Stage-I Results

The final evaluation is conducted on **CodaBench**. After generating your test predictions, package the submission files according to the competition requirements and upload them to:

https://www.codabench.org/competitions/15251/

> Note:
> - The official test results are only available through the CodaBench evaluation server.
> - Please make sure your submission file strictly follows the format required by the competition page.
> - Local validation can be used for debugging, but the leaderboard scores on CodaBench are the final results used for comparison.

---

## 4. Stage-II: Diffusion-Based Next Observation Generation

Stage-II trains a diffusion refinement model for next-observation generation. The model is a ControlNet-style latent diffusion model conditioned on:

- a source RGB image through the ControlNet hint pathway;
- a numeric pose condition, including heading and range, through a trainable pose encoder;
- optionally, a frozen RGB pose predictor used as an auxiliary pose-consistency loss.

### 4.1 Main Files

```text
step2_refine/
├── train_rgb_loss.py
├── train_rgb_condition_predictor.py
├── tutorial_dataset.py
├── cldm/
├── ldm/
├── cldm_v15_pose_hybrid.yaml
├── train_step2_example.sh
└── train_rgb_condition_predictor_example.sh
```

- `train_rgb_loss.py`: main Stage-II diffusion training script.
- `train_rgb_condition_predictor.py`: trains the frozen RGB pose predictor used by the auxiliary RGB pose-consistency loss.
- `tutorial_dataset.py`: PairUAV dataset loader for Stage-II training.
- `cldm/`: ControlNet and DreamNav model components.
- `ldm/`: latent diffusion model components.
- `cldm_v15_pose_hybrid.yaml`: Stage-II model configuration.
- `train_step2_example.sh`: example Stage-II training script.
- `train_rgb_condition_predictor_example.sh`: example RGB pose predictor training script.

### 4.2 External Files Required

The following files are not included in this repository:

- PairUAV dataset;
- base ControlNet checkpoint, e.g. `control_sd15_ini.ckpt`;
- Stage-I pose JSON, e.g. `step1_train_truepose.json` or predicted pose JSON;
- frozen RGB pose predictor checkpoint, e.g. `best.pt`.

### 4.3 Train RGB Pose Predictor

The RGB pose predictor takes a source RGB image and a target/generated RGB image as a 6-channel input pair, and predicts:

```text
[sin(heading), cos(heading), range / range_scale]
```

Train it with:

```bash
cd step2_refine/
bash train_rgb_condition_predictor_example.sh
```

The generated `best.pt` can be used as a frozen auxiliary model during Stage-II diffusion training.

### 4.4 Train Stage-II Diffusion Model

After preparing the dataset, base checkpoint, Stage-I pose JSON, and optional RGB predictor checkpoint, run:

```bash
cd step2_refine/
bash train_step2_example.sh
```

Please edit dataset paths, checkpoint paths, and output paths in the bash scripts before running.

### 4.5 Default Trainable Scope

For the default `train_mode=lora_control_decoder_hint` setting:

- the VAE is frozen;
- most pretrained diffusion backbone weights are frozen;
- the pose encoder is trainable;
- LoRA adapters are trained in selected ControlNet and UNet decoder linear layers;
- the ControlNet hint pathway is trainable;
- the RGB pose predictor is frozen and used only for auxiliary pose-consistency loss.

---

## 5. Files Not Included

Do not commit large datasets, checkpoints, or generated outputs:

```text
*.ckpt
*.pt
*.pth
*.safetensors
outputs*/
checkpoints/
pairUAV/
matches_data/
train_matches_data/
test_matches_data/
step1_*.json
lightning_logs/
wandb/
__pycache__/
*.pyc
```

## 🔗 Ecosystem

<p align="center"><i>Explore our ecosystem for UAV & Spatial Intelligence 🚁 </i></p >

### 🚁 UAV & Spatial Intelligence

<p align="center"><b>🎓 The University-1652 Family</b></p >

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <a href=" ">
          <h3>🎓</h3>
          <b>University-1652</b>
        </a >
        <br><sub>Multi-view Multi-source Benchmark<br>Ground · Drone · Satellite · ACM MM'20</sub>
        <br><br>
        <a href="https://github.com/layumi/University1652-Baseline"><img src="https://img.shields.io/github/stars/layumi/University1652-Baseline.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/wtyhub/MuseNet">
          <h3>🌦️</h3>
          <b>University-WX</b>
        </a >
        <br><sub>Multi-Weather Extension on the Fly<br>Pattern Recognition'24</sub>
        <br><br>
        <a href="https://github.com/wtyhub/MuseNet"><img src="https://img.shields.io/github/stars/wtyhub/MuseNet.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/MultimodalGeo/GeoText-1652">
          <h3>💬</h3>
          <b>GeoText-1652</b>
        </a >
        <br><sub>Dense Text Extension<br>ECCV'24</sub>
        <br><br>
        <a href="https://github.com/MultimodalGeo/GeoText-1652"><img src="https://img.shields.io/github/stars/MultimodalGeo/GeoText-1652.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
    </tr>
  </table>
</div>

<p align="center"><b>🚀 New Open-Source Releases</b></p >

<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <a href="https://github.com/YsongF/GeoFuse">
          <h3>🛰️</h3>
          <b>GeoFuse</b>
        </a >
        <br><sub>Road Maps as Free Geometric Priors </sub>
        <br><br>
        <a href="https://github.com/YsongF/GeoFuse"><img src="https://img.shields.io/github/stars/YsongF/GeoFuse.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/JT-Sun/UAVReason">
          <h3>🧠</h3>
          <b>UAVReason</b>
        </a >
        <br><sub>Aerial Scene Reasoning & Generation Benchmark</sub>
        <br><br>
        <a href="https://github.com/JT-Sun/UAVReason"><img src="https://img.shields.io/github/stars/JT-Sun/UAVReason.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/HaoDot/Video2BEV-Open">
          <h3>🗺️</h3>
          <b>Video2BEV</b>
        </a >
        <br><sub>Drone Video → Bird's-Eye-View</sub>
        <br><br>
        <a href="https://github.com/HaoDot/Video2BEV-Open"><img src="https://img.shields.io/github/stars/HaoDot/Video2BEV-Open.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
      <td align="center" width="25%">
        <a href="https://github.com/YaxuanLi-cn/PairUAV">
          <h3>🚁</h3>
          <b>PairUAV</b>
        </a >
        <br><sub>Paired UAV Data for Matching</sub>
        <br><br>
        <a href="https://github.com/YaxuanLi-cn/PairUAV"><img src="https://img.shields.io/github/stars/YaxuanLi-cn/PairUAV.svg?style=social&label=Star" alt="GitHub stars"></a >
      </td>
    </tr>
  </table>
</div>

---

<p align="center">
  ⭐ If you find our projects helpful, a <b>star</b> is the best support! ⭐
</p >