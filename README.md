# SIREN — Reproduction of Results
### Assignment 2 | Department of AI & Data Science | FAST-NUCES

> **Paper:** Implicit Neural Representations with Periodic Activation Functions  
> **Authors:** Sitzmann et al., NeurIPS 2020  
> **Official Repo:** https://github.com/vsitzmann/siren

---

## Group ID09

| Name | Roll Number | Assigned Experiments |
|------|-------------|----------------------|
| Valeena Afzal | 25I-8023 | Image, Audio, Video |
| Maryam Zafar | 25I-8033 | Poisson, SDF |
| Laiba Noor | 25I-8035 | Helmholtz, Wave Equation |

**Instructor:** Dr. Zohair Ahmed | **Program:** BS DS/AI

---

## What is SIREN?

SIREN (Sinusoidal Representation Network) is a neural network that uses **sine** as its activation function instead of ReLU. This allows it to naturally represent signals and their derivatives, making it ideal for solving partial differential equations (PDEs) and fitting complex continuous signals.

```
φᵢ(xᵢ) = sin(ω₀ · (Wᵢxᵢ + bᵢ))     where ω₀ = 30
```

---

## Repository Structure

```
siren/                          ← Official repo (cloned)
│
├── experiment_scripts/
│   ├── train_helmholtz.py      ← Laiba's experiment 1
│   ├── train_wave_equation.py  ← Laiba's experiment 2
│   ├── train_img.py            ← Valeena's experiment
│   ├── train_audio.py          ← Valeena's experiment
│   ├── train_video.py          ← Valeena's experiment
│   ├── train_poisson_*.py      ← Maryam's experiments
│   └── train_sdf.py            ← Maryam's experiment
│
├── dataio.py                   ← Dataset classes
├── modules.py                  ← SIREN architecture
├── loss_functions.py           ← PDE loss functions
├── training.py                 ← Training loop
├── diff_operators.py           ← Gradient/Laplacian operators
├── utils.py                    ← TensorBoard summaries
└── torchmeta/                  ← Bundled (do NOT pip install)

notebooks/
├── DL_assignment_2_part_1.ipynb   ← Helmholtz experiment notebook
└── DL_assignment_2_part_2.ipynb   ← Wave equation experiment notebook

results/
├── helmholtz_loss_curve.png
├── helmholtz_wavefield.png
├── wave_loss_curve.png
└── wave_wavefield.png
```

---

## Setup (Google Colab — Recommended)

**Step 1 — Clone repo**
```python
!git clone https://github.com/vsitzmann/siren.git
%cd siren
```

**Step 2 — Install missing libraries**
```python
!pip install scikit-video cmapy configargparse
```

**Step 3 — Apply bug fixes**
```python
import subprocess

# Fix 1: Class name typo in utils.py
subprocess.run(['sed', '-i',
    's/NeuralProcessImplicit2DHypernetBVP/NeuralProcessImplicit2DHypernet/g',
    'utils.py'])

# Fix 2: Comment out summary_fn to prevent CUDA OOM (wave equation)
subprocess.run(['sed', '-i',
    's/summary_fn(model, model_input, gt, model_output, writer, total_steps)/#summary_fn(model, model_input, gt, model_output, writer, total_steps)/g',
    'training.py'])

# Fix 3: Reduce wave model size for 14GB GPU
subprocess.run(['sed', '-i',
    's/hidden_features=512/hidden_features=256/g',
    'experiment_scripts/train_wave_equation.py'])

print("All fixes applied!")
```

---

## Running Experiments

### Helmholtz Equation (Laiba)
```python
!python experiment_scripts/train_helmholtz.py \
    --experiment_name helmholtz_repro \
    --logging_root ./logs \
    --num_epochs 5000
```

### Wave Equation (Laiba)
```python
!python experiment_scripts/train_wave_equation.py \
    --experiment_name wave_repro \
    --logging_root ./logs \
    --num_epochs 5000 \
    --batch_size 3000
```

### Image Fitting (Valeena)
```python
!python experiment_scripts/train_img.py \
    --model_type sine \
    --experiment_name image_repro \
    --logging_root ./logs
```

### Audio Fitting (Valeena)
```python
!python experiment_scripts/train_audio.py \
    --model_type sine \
    --wav_path data/gt_bach.wav \
    --experiment_name audio_repro \
    --logging_root ./logs
```

### Poisson Equation (Maryam)
```python
!python experiment_scripts/train_poisson_grad_img.py \
    --experiment_name poisson_repro \
    --logging_root ./logs
```

---

## Libraries Used

| Library | Version | How Installed | Purpose |
|---------|---------|---------------|---------|
| PyTorch | Pre-installed | Colab default | Core framework |
| NumPy | 2.0.2 | Colab default | Array operations |
| Matplotlib | 3.10.0 | Colab default | Plotting |
| SciPy | 1.16.3 | Colab default | Differential operators |
| TensorBoard | 2.19.0 | Colab default | Training monitoring |
| scikit-video | 1.1.11 | `pip install scikit-video` | Video I/O |
| cmapy | 0.6.6 | `pip install cmapy` | Colormaps |
| opencv-python | 4.13.0.92 | Colab default | Image processing |
| scikit-image | 0.25.2 | Colab default | Image utilities |
| configargparse | 1.7.5 | `pip install configargparse` | Argument parsing |
| torchmeta | bundled | Local repo folder | Hypernetworks |

> **Note:** Do NOT `pip install torchmeta` — it is incompatible with modern PyTorch. Use the bundled `./torchmeta/` folder instead.

---

## Results

### Helmholtz Equation

| Metric | Paper | Ours |
|--------|-------|------|
| Training Steps | 50,000 | 5,000 |
| Initial Loss | — | 5,581,349 |
| Final Loss | Converged | 175,564 |
| Loss Reduction | ~96%+ | 96.9% |


### Wave Equation

| Metric | Paper | Ours |
|--------|-------|------|
| Training Steps | 100,000 | 5,000 |
| Initial Loss | — | 85,955,296 |
| Final Loss | Converged | 16,236 |
| Loss Reduction | — | 99.98% |


---

## Known Issues & Fixes

### 1. `AttributeError: NeuralProcessImplicit2DHypernetBVP`
**Cause:** Typo in `utils.py` — class does not exist  
**Fix:** Replace with `NeuralProcessImplicit2DHypernet` (see setup Step 3)

### 2. `CUDA Out of Memory` — Wave Equation
**Cause:** `summary_fn()` computes Jacobians for visualization, consuming 13+ GB  
**Fix:** Comment out `summary_fn()` in `training.py` (see setup Step 3)  
**Reference:** GitHub Issue [#20](https://github.com/vsitzmann/siren/issues/20)

### 3. `torchmeta` pip install fails
**Cause:** All versions require `torch < 1.10`, incompatible with modern PyTorch  
**Fix:** Use the bundled `./torchmeta/` folder — do not pip install

### 4. `FileExistsError` when re-running experiments
**Cause:** Log folder already exists from a previous run  
**Fix:** `!rm -rf ./logs/<experiment_name>` before re-running

### 5. Large Helmholtz loss values (millions)
**Cause:** Loss is summed over all sampled points, not averaged  
**Note:** This is expected and confirmed by paper authors in GitHub Issue [#7](https://github.com/vsitzmann/siren/issues/7)

---

## Hardware

- **Platform:** Google Colab (free tier)
- **GPU:** NVIDIA T4 (14.56 GB VRAM)
- **Training time:** ~63 min (Helmholtz, 5000 steps) | ~67 min (Wave, 5000 steps)

---

## Citation

```bibtex
@inproceedings{sitzmann2019siren,
    author    = {Sitzmann, Vincent and Martel, Julien N.P. and
                 Bergman, Alexander W. and Lindell, David B. and Wetzstein, Gordon},
    title     = {Implicit Neural Representations with Periodic Activation Functions},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2020}
}
```

---

## References

1. Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions," NeurIPS 2020
2. Official SIREN GitHub: https://github.com/vsitzmann/siren
3. GitHub Issue #7 (Helmholtz loss): https://github.com/vsitzmann/siren/issues/7
4. GitHub Issue #20 (Wave OOM): https://github.com/vsitzmann/siren/issues/20
