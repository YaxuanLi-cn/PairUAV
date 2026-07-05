<h1 align="center">Last-Meter Precision Navigation for UAVs: A Diffusion-Refined\\ Aerial Visual Servoing Approach</h1>


<p align="center">
  <!-- Dataset -->
  <a href="https://huggingface.co/datasets/YaxuanLi/UAVM_2026_test">
    <img src="https://img.shields.io/badge/Dataset-HF%20Data-d8b04c?style=for-the-badge" alt="Dataset">
  </a>

  <!-- Workshop -->
  <a href="https://www.zdzheng.xyz/ACMMM2026Workshop-UAV/">
    <img src="https://img.shields.io/badge/Workshop-Page-d46a5a?style=for-the-badge" alt="Workshop">
  </a>

  <!-- Email -->
  <a href="yaxuanli.cn@gmail.com">
    <img src="https://img.shields.io/badge/Email-YaxuanLi-6b6b6b?style=for-the-badge" alt="Email">
  </a>
</p>

<hr />

## Project Structure

```
UAVM_2026/
├── models/   
├── pairUAV/
│   ├── data_process.sh 
│   └── University-Release.zip
├── baseline/                  
│   ├── SuperGlue/   
│   ├── train.py      
└── └── run.sh             
```

---

## 1. Environment Setup

Create a unified conda environment for our baseline:

```bash
conda create -n uavm python=3.9
conda activate uavm

# Install PyTorch with CUDA support
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install all dependencies
pip install -r requirements.txt

# Download pretrained models
huggingface-cli download Ramos-Ramos/dino-resnet-50 --local-dir models/dino_resnet
```

---

## 2. Data Preparation

### 2.1 Download University-1652 Dataset

Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request (Usually I will reply you in 5 minutes). You may use the [request template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

### 2.2 Download and Process PairUAV Dataset

Download and process the PairUAV dataset:

```bash
cd pairUAV/
bash data_process.sh
cd ..
```

This script downloads the dataset from HuggingFace and extracts train/test/tours data to the `pairUAV/` directory.

## 3. SuperGlue-Based Baseline

### 3.1 Run SuperGlue Feature Matching

First, perform feature matching on image pairs:

```bash
cd baseline/SuperGlue

# Running everything takes about 6 hours in total. You can also directly download the results we have already generated using the following method.
bash download_results.sh
cd ..

# Or Run feature matching
python gen_test_pairs.py
bash run_train.sh
bash run_test.sh
cd ..
```

This generates matching results in `train_matches_data/` and `test_matches_data/`.

### 3.2 Train Model

```bash
bash run.sh
cd ..
```

### 3.3 Evaluate Results

The final evaluation is conducted on **CodaBench**.  
After generating your test predictions, please package the submission files according to the competition requirements and upload them to:

**https://www.codabench.org/competitions/15251/**

> **Note**
> - The official test results are only available through the CodaBench evaluation server.
> - Please make sure your submission file strictly follows the format required by the competition page.
> - Local validation can be used for debugging, but the leaderboard scores on CodaBench are the final results used for comparison.

---
