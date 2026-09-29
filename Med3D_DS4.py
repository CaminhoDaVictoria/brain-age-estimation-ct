import copy
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


'''
Convert y_true and y_pred to NumPy arrays if they are provided
as lists or other array-like formats.

This ensures faster and more reliable numerical computations.

ravel() flattens the input into a one-dimensional array:
e.g., [[1, 2], [3, 4]] -> [1, 2, 3, 4]

This guarantees that both arrays have a consistent shape for
error metric calculations.
'''
def compute_core_metrics(y_true, y_pred):
   
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Compute the Mean Absolute Error (MAE), i.e., the average absolute difference between predictions and ground-truth values
    mae = float(mean_absolute_error(y_true, y_pred))

    # BAG (Brain Age Gap): prediction error for each individual sample, computed as predicted age minus chronological age (Pred - Actual)
    bag = y_pred - y_true


    '''
    Mean Brain Age Gap (BAG).
    Represents the average BAG across all samples and indicates
    whether the model systematically overestimates or underestimates age.
    Positive values suggest overestimation, whereas negative values
    suggest underestimation.
    '''
    bag_mean = float(np.mean(bag))

    '''
    R² score: measures how well the predictions explain the variance
    of the ground-truth values.

    Returns NaN when only a single sample is available, since R² is
    not defined for datasets containing only one observation.
    '''
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')

    '''
    Spearman rank correlation: measures whether the ranking of the
    predicted values corresponds to the ranking of the ground-truth values.

    rho : Spearman correlation coefficient
        Indicates the strength and direction of the monotonic relationship.

    p_value : Statistical significance of rho
        Represents the probability of observing the correlation by chance.
    '''
    rho, pval = spearmanr(y_true, y_pred)

    '''
    Safety check: if rho or p_value is NaN, explicitly replace it
    with float('nan').
    This ensures consistent handling of invalid or undefined correlation
    results and prevents downstream issues during metric aggregation or
    serialization.
    '''
    rho = float(rho) if not np.isnan(rho) else float('nan')
    pval = float(pval) if not np.isnan(pval) else float('nan')

    # Return all computed metrics together in a single dictionary
    return {"MAE": mae, "BAG_mean": bag_mean, "R2": r2, "Spearman_rho": rho, "Spearman_p": pval}

'''
Convert the inputs to NumPy arrays again and flatten them to
one-dimensional vectors.
This ensures that both arrays have a consistent shape for
error metric calculations and other element-wise operations.
'''
def compute_extra_metrics(y_true, y_pred, thresholds=(3,5,10)):

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Absolute prediction error for each individual sample
    abs_err = np.abs(y_pred - y_true)

    '''
    Brain Age Gap (BAG) for each individual sample.

    This metric is important for assessing how accurately the model
    estimates chronological age on a per-subject basis.

    It can also help identify whether certain patients or subgroups
    systematically exhibit positive or negative Brain Age Gaps,
    which may indicate potential biological, clinical, or demographic
    effects.
    '''
    bag = y_pred - y_true

    '''
    Compute additional evaluation metrics:

    RMSE (Root Mean Squared Error)
        Penalizes large prediction errors more strongly than MAE.

    MedianFE (Median Absolute Error)
        Median of the absolute prediction errors, providing a robust
        measure that is less sensitive to outliers.

    Standard Deviation (SD)
        Quantifies the variability of the prediction errors across samples.

    Bias
        Mean signed prediction error (Pred - Actual), indicating whether
        the model systematically overestimates or underestimates age.
    '''
    res = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedianAE": float(np.median(abs_err)),
        "SD_BAG": float(np.std(bag, ddof=1)),
        "MeanBias": float(np.mean(bag))
    }

    '''
    Compute the percentage of predictions that fall within predefined
    error thresholds.

    For example, pct_within_3 represents the percentage of predictions
    that are within ±3 years of the true age.
    '''
    for t in thresholds:
        res[f"pct_within_{t}"] = float((abs_err <= t).mean() * 100.0)
    return res

'''
Computes a confidence interval for the Mean Absolute Error (MAE)
using bootstrapping.

Bootstrapping is a resampling technique in which random samples are
drawn with replacement to estimate statistical uncertainty.

The resulting confidence interval reflects the sampling uncertainty
within the validation dataset. It should not be interpreted as a
substitute for external validation on an independent dataset.
'''
def bootstrap_ci_mae(y_true, y_pred, n_bootstrap=1000, alpha=0.05, random_state=0):
    rng = np.random.default_rng(random_state)   # Random number generator used to ensure reproducibility
    y_true = np.asarray(y_true)                
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    # Array containing the MAE values from all bootstrap samples
    maes = np.empty(n_bootstrap, dtype=float)

    # Iterate over all bootstrap samples
    for i in range(n_bootstrap):
        '''
        Draw random indices from 0 to n-1 with replacement.

        rng.integers(0, n, n) generates an array of n random integers
        between 0 and n-1.

        Sampling with replacement means that the same index can appear
        multiple times in the bootstrap sample.

        Example:
            n = 5
            y_true = [10, 12, 14, 16, 18]

            idx = [2, 0, 3, 2, 4]

        This simulates a new dataset drawn from the original observations,
        which is the core idea behind bootstrapping.
        '''
        idx = rng.integers(0, n, n)
       
        '''
            Apply the bootstrap sample to the ground-truth and predicted values.

            y_true[idx] and y_pred[idx] create new arrays containing exactly
            the randomly selected samples.

            Example:
                y_true[idx] = [14, 10, 16, 14, 18]

            and y_pred[idx] contains the corresponding predictions.

            Compute the Mean Absolute Error (MAE) for this bootstrap sample.

            mean_absolute_error() calculates the absolute difference between
            predictions and ground-truth values and then computes their average.

            This provides an estimate of how well the model performs on the
            resampled dataset.
        '''
        maes[i] = mean_absolute_error(y_true[idx], y_pred[idx])

    # Determine the lower and upper percentiles of the bootstrap distribution to construct the confidence interval
    lower = float(np.percentile(maes, 100 * (alpha/2)))         # lower boundary
    upper = float(np.percentile(maes, 100 * (1-alpha/2)))       # upper boundary
    return lower, upper

# ---- Reproducibility / Set seed ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


'''
Generates a filesystem-safe filename from a plot title.

This ensures that plots are saved correctly in the designated
output directory and prevents invalid path or filename issues
caused by special characters.
'''
def safe_filename(text):
    
    text = str(text).strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:180]

'''
Saves the current Matplotlib figure as both PNG and PDF.

PNG is convenient for quick visualization and sharing, while PDF
provides a high-quality, scalable format suitable for reports,
publications, and documentation.
'''

def save_current_plot(save_dir, title):
   
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(title)
    plt.savefig(save_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.savefig(save_dir / f"{stem}.pdf", bbox_inches="tight")


# ------------------------------------------------
# Scatter plot: Predicted vs Actual
# ------------------------------------------------
def plot_pred_vs_actual_dual(y_true, y_pred_baseline, y_pred_model, save_dir,
                             title="Predicted vs Actual: Baseline vs Model"):
    y_true = np.asarray(y_true)
    y_pred_baseline = np.asarray(y_pred_baseline)
    y_pred_model = np.asarray(y_pred_model)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred_baseline, alpha=0.6, label="Baseline untrained", marker="o")
    plt.scatter(y_true, y_pred_model, alpha=0.6, label="Model trained", marker="x")

    mn = np.nanmin([y_true.min(), y_pred_baseline.min(), y_pred_model.min()])
    mx = np.nanmax([y_true.max(), y_pred_baseline.max(), y_pred_model.max()])
    plt.plot([mn, mx], [mn, mx], "--", color="gray")

    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_current_plot(save_dir, title)
    plt.close()


# ------------------------------------------------
# Residuals vs Actual
# ------------------------------------------------
def plot_residuals_vs_actual_dual(y_true, y_pred_baseline, y_pred_model, save_dir,
                                  title="Residuals vs Actual: Baseline vs Model"):
    y_true = np.asarray(y_true)
    residuals_baseline = np.asarray(y_pred_baseline) - y_true
    residuals_model = np.asarray(y_pred_model) - y_true

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, residuals_baseline, alpha=0.6, label="Baseline untrained", marker="o")
    plt.scatter(y_true, residuals_model, alpha=0.6, label="Model trained", marker="x")
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Actual Age")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_current_plot(save_dir, title)
    plt.close()


# ---------------------------------------------------
# Histogram of the Brain-Age-Gap (BAG) distribution
# ---------------------------------------------------
def plot_bag_hist_dual(y_true, y_pred_baseline, y_pred_model, save_dir,
                       title="Histogram of BAG: Baseline vs Model", bins=30):
    y_true = np.asarray(y_true)
    bag_baseline = np.asarray(y_pred_baseline) - y_true
    bag_model = np.asarray(y_pred_model) - y_true

    plt.figure(figsize=(6, 4))
    plt.hist(bag_baseline, bins=bins, alpha=0.5, label="Baseline untrained")
    plt.hist(bag_model, bins=bins, alpha=0.5, label="Model trained")
    plt.xlabel("Brain-Age Gap (years)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_current_plot(save_dir, title)
    plt.close()


# ------------------------------------------------
# Bland-Altman Plot
# ------------------------------------------------
def plot_bland_altman_dual(y_true, y_pred_baseline, y_pred_model, save_dir,
                           title="Bland-Altman: Baseline vs Model"):
    y_true = np.asarray(y_true)
    y_pred_baseline = np.asarray(y_pred_baseline)
    y_pred_model = np.asarray(y_pred_model)

    plt.figure(figsize=(6, 4))
    mean_vals_b = (y_pred_baseline + y_true) / 2.0
    diff_b = y_pred_baseline - y_true
    mean_vals_m = (y_pred_model + y_true) / 2.0
    diff_m = y_pred_model - y_true

    plt.scatter(mean_vals_b, diff_b, alpha=0.5, label="Baseline")
    plt.scatter(mean_vals_m, diff_m, alpha=0.5, label="Model")

    mean_diff = float(np.mean(diff_m))
    sd_diff = float(np.std(diff_m, ddof=1)) if len(diff_m) > 1 else 0.0
    plt.axhline(mean_diff, color="red", label="Model mean BAG")
    plt.axhline(mean_diff - 1.96 * sd_diff, linestyle="--", color="gray")
    plt.axhline(mean_diff + 1.96 * sd_diff, linestyle="--", color="gray")

    plt.xlabel("Mean of Predicted and Actual Age")
    plt.ylabel("Difference (Predicted - Actual)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_current_plot(save_dir, title)
    plt.close()


'''
Saves the loss and MAE curves for a model over the course of training.

These learning curves help assess whether the model is still improving,
has converged, or is beginning to overfit the training data.
'''
def plot_training_curves(history_df, save_dir, model_name):

    if history_df.empty:
        return

    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train weighted loss")
    plt.plot(history_df["epoch"], history_df["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"{model_name} Training Curves")
    plt.legend()
    plt.tight_layout()
    save_current_plot(save_dir, f"{model_name}_training_curves")
    plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


'''
Applies 3D data augmentation during training for brain age prediction
volumes.

This function is used exclusively within the training dataset.

Validation and test datasets remain unchanged to ensure that
evaluation metrics are unbiased, reproducible, and directly
comparable across experiments.
'''
def augment_volume_3d(volume, flip_prob=0.5, max_rotation=5.0, noise_std=0.01):
  
    vol = np.asarray(volume, dtype=np.float32).copy()

    # Left-right flip as a simple and robust augmentation strategy
    if np.random.rand() < flip_prob:
        vol = np.flip(vol, axis=2).copy()

    # Small in-plane rotation applied per volume
    if max_rotation and max_rotation > 0:
        angle = np.random.uniform(-max_rotation, max_rotation)
        for z in range(vol.shape[0]):
            vol[z] = ndi.rotate(vol[z], angle, reshape=False, order=1, mode="nearest")

    # Mild Gaussian noise augmentation
    if noise_std and noise_std > 0:
        vol = vol + np.random.normal(0, noise_std, size=vol.shape).astype(np.float32)

    return np.clip(vol, 0.0, 1.0).astype(np.float32)


'''
Computes sample weights based on the age distribution.

If younger patients are more common than older patients in the dataset,
underrepresented age groups receive higher weights in the loss function.

This reduces bias toward the dominant younger age group and encourages
the model to learn age-related patterns more evenly across the entire
age range.
'''
def compute_age_sample_weights(ages, n_bins=4, min_weight=1.0, max_weight=3.0):
 
    ages = np.asarray(ages, dtype=np.float32)
    if len(ages) == 0:
        return np.array([], dtype=np.float32)

    # Quantile-based bins are robust because they automatically adapt to the age distribution of the specific cohort
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ages, quantiles))
    if len(edges) <= 2:
        return np.ones_like(ages, dtype=np.float32)

    bin_ids = np.digitize(ages, edges[1:-1], right=True)
    counts = np.bincount(bin_ids, minlength=len(edges) - 1).astype(np.float32)
    inv_freq = 1.0 / np.maximum(counts, 1.0)
    weights = inv_freq[bin_ids]

    '''
    Normalize the sample weights so that the average weight is
    approximately 1, then apply clipping to limit extreme values.

    Normalization keeps the overall loss scale stable and prevents
    artificial changes in optimization dynamics.

    Clipping ensures that very rare age groups receive increased
    importance without completely dominating the loss function.
    '''
    weights = weights / np.mean(weights)
    weights = np.clip(weights, min_weight, max_weight)
    return weights.astype(np.float32)


# ---- Custom dataset for 3D volumes ----
'''
volumes : list
    List of 3D volumes with shape (D, H, W).

ages : list or np.ndarray
    Chronological age corresponding to each volume.

patient_ids : list, optional
    Patient identifiers for subsequent CSV exports or analyses.

volume_names : list, optional
    Volume filenames for subsequent CSV exports or analyses.

sample_weights : list or np.ndarray, optional
    Per-sample weights used for age-weighted loss functions.

augment : bool
    If True, training-time data augmentation is applied.
    This should only be enabled for training data.

transform : callable, optional
    Custom transformation pipeline applied to the volumes.
'''
class VolumeDataset(Dataset):
    def __init__(self, volumes, ages, patient_ids=None, volume_names=None, sample_weights=None, augment=False, transform=None):
      
        self.volumes = volumes
        self.ages = np.asarray(ages, dtype=np.float32)
        self.patient_ids = list(patient_ids) if patient_ids is not None else list(range(len(volumes)))
        self.volume_names = list(volume_names) if volume_names is not None else [str(i) for i in range(len(volumes))]
        self.sample_weights = (
            np.asarray(sample_weights, dtype=np.float32)
            if sample_weights is not None else np.ones(len(volumes), dtype=np.float32)
        )
        self.augment = augment
        self.transform = transform
        self.target_shape = tuple(np.max([np.asarray(v).shape for v in volumes], axis=0).astype(int))

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        arr = np.asarray(self.volumes[idx], dtype=np.float32)

        if self.augment:
            arr = augment_volume_3d(arr)

        if self.transform is not None:
            arr = self.transform(arr)

        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]  # (1,D,H,W)

        D, H, W = arr.shape[1:]
        target_D, target_H, target_W = self.target_shape
        pad_D = max(target_D - D, 0)
        pad_H = max(target_H - H, 0)
        pad_W = max(target_W - W, 0)
        if pad_D or pad_H or pad_W:
            arr = np.pad(arr, ((0, 0), (0, pad_D), (0, pad_H), (0, pad_W)), mode="constant")

        tensor = torch.from_numpy(arr.copy()).float()
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return tensor, age, weight, self.patient_ids[idx], self.volume_names[idx]


# ---- Custom Collate Function ----
def collate_stack(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    ws = [b[2] for b in batch]
    patient_ids = [b[3] for b in batch]
    volume_names = [b[4] for b in batch]
    return torch.stack(xs, 0), torch.stack(ys, 0), torch.stack(ws, 0), patient_ids, volume_names


'''
Provides a unified model output interface for training,
validation, and baseline evaluation.

Softplus is used instead of ReLU as the final positive activation
function. Unlike ReLU, Softplus is smooth and avoids hard zero-valued
regions, which is generally advantageous for age regression tasks.
'''
def model_forward_age(model, xb):

    out = model(xb)
    if out.dim() == 5:
        out = out.mean(dim=[2, 3, 4])
    out = out.view(out.shape[0], -1)[:, 0]
    return F.softplus(out)


'''
Freezes Batch Normalization statistics by keeping BatchNorm layers
in evaluation mode.

When training with batch_size=1, BatchNorm statistics computed from
individual batches are highly unstable. If BatchNorm layers continue
to update their running statistics during training, these running
estimates gradually drift away from the stable pretrained values.

During evaluation, BatchNorm then normalizes using mismatched
statistics, which can cause predictions to become unstable or even
explode, resulting in validation MAE values orders of magnitude
larger than the training MAE despite apparently stable training.

By freezing BatchNorm, both training and validation use the same
stable normalization statistics, leading to more reliable and
consistent predictions.
'''
def set_bn_eval(model):
   
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


'''
Predictions are constrained to a physiologically plausible age range.

The primary safeguard against exploding predictions is set_bn_eval(),
which stabilizes Batch Normalization behavior.

This clipping step serves only as an additional safety measure,
preventing isolated outliers from disproportionately influencing
evaluation metrics.
'''
PRED_CLIP_RANGE = (0.0, 120.0)


# ---- Performs batched inference to generate model predictions ----
def batch_predict_from_loader(model, loader, device=device, clip_range=None):
    model.eval()
    preds, trues, patient_ids, volume_names = [], [], [], []
    with torch.no_grad():
        for xb, yb, wb, ids, names in loader:
            xb = xb.to(device)
            out = model_forward_age(model, xb)
            preds.extend(out.cpu().numpy().tolist())
            trues.extend(yb.numpy().tolist())
            patient_ids.extend(ids)
            volume_names.extend(names)

    preds = np.array(preds)
    if clip_range is not None:
        preds = np.clip(preds, clip_range[0], clip_range[1])
    return np.array(trues), preds, patient_ids, volume_names

'''
Weighted Mean Absolute Error (MAE).

This loss function allows older or underrepresented age groups
to contribute more strongly to the optimization process through
sample-specific weights.

By increasing the influence of rare age ranges, the model is
encouraged to learn age-related patterns more evenly across the
entire age distribution.
'''

def weighted_mae_loss(pred, target, weights):
    
    loss = torch.abs(pred - target.float())
    weighted = loss * weights.float()
    return weighted.sum() / torch.clamp(weights.float().sum(), min=1e-6)


PROJECT_DIR = Path("") # project path
PROCESSED_DATA_DIR = PROJECT_DIR / "processed_data"

DATASET_ID = "DS4"
DATASET_ID_LOWER = DATASET_ID.lower()

volume_dir = PROCESSED_DATA_DIR / f"{DATASET_ID}_Volumes"
csv_path = volume_dir / f"patient_metadata_{DATASET_ID_LOWER}.csv"

if not volume_dir.exists():
    raise FileNotFoundError(f"Volume folder not found: {volume_dir}")
if not csv_path.exists():
    raise FileNotFoundError(f"Meta csv not found: {csv_path}")

print("Dataset:", DATASET_ID)
print("Volume folder:", volume_dir)
print("Meta-CSV:", csv_path)

# Required columns in the metadata CSV
csv_id_col = "Folder" # align with labels
csv_age_col = "PatientAge" # align with labels

# ---- Load volumes and match them with metadata ----
vol_paths = sorted(volume_dir.glob("*.npy"))
print("Found volume files:", len(vol_paths))

if len(vol_paths) == 0:
    raise FileNotFoundError(f"No .npy-volumes found in: {volume_dir}")

df = pd.read_csv(csv_path)
df[csv_id_col] = df[csv_id_col].astype(str)
folder_to_age = dict(zip(df[csv_id_col], df[csv_age_col]))

volumes = []
vol_ids = []
vol_names = []
used_paths = []
missing_labels = []

for path in vol_paths:
    name = path.stem
    # Expected naming convention: volume_10_ds9 -> patient ID = 10
    parts = name.split("_")
    if len(parts) < 2:
        print(f"Skip file with unexpected name: {name}")
        continue

    pid = parts[1]
    if pid not in folder_to_age:
        missing_labels.append(pid)
        continue

    volumes.append(np.load(path).astype(np.float32))
    vol_ids.append(pid)
    vol_names.append(name)
    used_paths.append(str(path))

if missing_labels:
    print("Warning: For these IDs labels are missing: ", sorted(set(missing_labels))[:20])

if len(volumes) == 0:
    raise ValueError("No volumes could be matched with labels from the csv.")

# Sort entries by their numeric identifier whenever possible to ensure deterministic and reproducible processing
def _sort_key(pid):
    try:
        return int(pid)
    except Exception:
        return str(pid)

sorted_idx = sorted(range(len(vol_ids)), key=lambda i: _sort_key(vol_ids[i]))
volumes = [volumes[i] for i in sorted_idx]
vol_ids = [vol_ids[i] for i in sorted_idx]
vol_names = [vol_names[i] for i in sorted_idx]
used_paths = [used_paths[i] for i in sorted_idx]
y = np.array([float(folder_to_age[str(pid)]) for pid in vol_ids], dtype=np.float32)

print("Loaded and matched volumes: ", len(volumes))
print("Range of age: ", float(np.min(y)), "to", float(np.max(y)))


# Add the path to the local ResNet implementation directory
MEDICALNET_MODEL_DIR = PROJECT_DIR / "MedicalNet" / "models"
RESNET_FALLBACK_DIR = PROJECT_DIR

if (MEDICALNET_MODEL_DIR / "resnet.py").exists():
    resnet_dir = MEDICALNET_MODEL_DIR
elif (RESNET_FALLBACK_DIR / "resnet.py").exists():
    resnet_dir = RESNET_FALLBACK_DIR
else:
    raise FileNotFoundError(
        f"resnet.py whether in {MEDICALNET_MODEL_DIR} nor in {RESNET_FALLBACK_DIR} found."
    )

resnet_dir_str = str(resnet_dir)
if resnet_dir_str not in sys.path:
    sys.path.insert(0, resnet_dir_str)
print("ResNet-Code:", resnet_dir)

# Import models
from resnet import ResNet, BasicBlock, Bottleneck


# -------------------------
# Train/Validation Split
# -------------------------
batch_size_default = 2
test_size = 0.20

'''
Optional age stratification to ensure that younger and older
patients are distributed as evenly as possible between the
training and validation sets.

If the cohort is too small to support reliable stratification,
the code automatically falls back to a standard random split.
'''
try:
    age_bins = pd.qcut(y, q=min(4, len(np.unique(y))), duplicates="drop", labels=False)
    stratify_arg = age_bins if len(np.unique(age_bins)) > 1 else None
except Exception:
    stratify_arg = None

split = train_test_split(
    volumes,
    y,
    vol_ids,
    vol_names,
    test_size=test_size,
    random_state=SEED,
    shuffle=True,
    stratify=stratify_arg
)
train_vols, val_vols, y_train, y_val, train_ids, val_ids, train_names, val_names = split
print("Train samples:", len(train_vols), "Val samples:", len(val_vols))

'''
Compute sample weights using only the training split.

This prevents information from the validation age distribution
from leaking into the training loss, ensuring a clean separation
between training and evaluation data.
'''
train_weights = compute_age_sample_weights(y_train, n_bins=4, min_weight=1.0, max_weight=3.0)
val_weights = np.ones(len(y_val), dtype=np.float32)

# Augmentation only activated for training dataset
train_dataset = VolumeDataset(
    train_vols,
    y_train,
    patient_ids=train_ids,
    volume_names=train_names,
    sample_weights=train_weights,
    augment=True
)
val_dataset = VolumeDataset(
    val_vols,
    y_val,
    patient_ids=val_ids,
    volume_names=val_names,
    sample_weights=val_weights,
    augment=False
)


# -------------------------
# Prepare models
# -------------------------
block_dict = {"basic": BasicBlock, "bottleneck": Bottleneck}
PRETRAIN_DIR = PROJECT_DIR / "MedicalNet_pytorch_files2" / "pretrain"

model_files = {
    "resnet10": (PRETRAIN_DIR / "resnet_10.pth", "basic", [1, 1, 1, 1]),
    "resnet10_23datasets": (PRETRAIN_DIR / "resnet_10_23dataset.pth", "basic", [1, 1, 1, 1]),
    "resnet18": (PRETRAIN_DIR / "resnet_18.pth", "basic", [2, 2, 2, 2]),
    "resnet18_23datasets": (PRETRAIN_DIR / "resnet_18_23dataset.pth", "basic", [2, 2, 2, 2]),
    "resnet34": (PRETRAIN_DIR / "resnet_34.pth", "basic", [3, 4, 6, 3]),
    "resnet34_23datasets": (PRETRAIN_DIR / "resnet_34_23dataset.pth", "basic", [3, 4, 6, 3]),
    "resnet50": (PRETRAIN_DIR / "resnet_50.pth", "bottleneck", [3, 4, 6, 3]),
    "resnet50_23datasets": (PRETRAIN_DIR / "resnet_50_23dataset.pth", "bottleneck", [3, 4, 6, 3]),
    "resnet101": (PRETRAIN_DIR / "resnet_101.pth", "bottleneck", [3, 4, 23, 3]),
    "resnet152": (PRETRAIN_DIR / "resnet_152.pth", "bottleneck", [3, 8, 36, 3]),
    "resnet200": (PRETRAIN_DIR / "resnet_200.pth", "bottleneck", [3, 24, 36, 3]),
}

missing_checkpoints = [str(pth) for pth, _, _ in model_files.values() if not Path(pth).exists()]
if missing_checkpoints:
    raise FileNotFoundError("Missing MedicalNet-Checkpoints:\n" + "\n".join(missing_checkpoints))

# -------------------------
# Hyperparameter
# -------------------------
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 12
SAVE_DIR = PROJECT_DIR / DATASET_ID / "trained_med3d_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = SAVE_DIR / "plots"
HEATMAP_DIR = SAVE_DIR / "heatmaps"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

RUN_MODEL_NAMES = list(model_files.keys())
N_HEATMAPS_PER_MODEL = 3 # Specify the number of heatmaps to generate. By default, three heatmaps are created using the first three samples from the validation set

results_all = {}
baseline_results_all = {}
all_prediction_tables = []


# ==========================================
# Helper: Build models and load checkpoint
# ==========================================
def build_med3d_model(model_name, block_type, layers):
    D, H, W = volumes[0].shape
    block_cls = block_dict[block_type]
    model = ResNet(
        block=block_cls,
        layers=layers,
        sample_input_D=D,
        sample_input_H=H,
        sample_input_W=W,
        num_seg_classes=1,
        shortcut_type="B",
        no_cuda=(device.type == "cpu")
    ).to(device)
    return model

'''
Loads MedicalNet pretrained weights if a valid checkpoint path
is provided.

Non-matching keys are tolerated because the regression head
often differs from the original pretrained architecture.
'''
def load_pretrained_if_available(model, pth_file):
   
    pth = Path(str(pth_file).strip())
    if not pth.exists():
        print("No pretrained checkpoint loaded; path is missing:", pth_file)
        return False

    ckpt = torch.load(pth, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    new_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(new_state, strict=False)
    print("Pretrained loaded:", pth)
    print("Missing keys:", len(incompatible.missing_keys), "Unexpected keys:", len(incompatible.unexpected_keys))
    return True


# ============================================================
# Helper: Evaluate model
# ============================================================
def evaluate_model(model, loader, device, clip_range=None):
    trues, preds, patient_ids, volume_names = batch_predict_from_loader(
        model,
        loader,
        device=device,
        clip_range=clip_range
    )
    return trues, preds, patient_ids, volume_names


def save_metrics_csv(metrics, path):
    pd.DataFrame([metrics]).to_csv(path, index=False)


def make_prediction_table(model_name, split_name, trues, preds, patient_ids, volume_names, pred_col):
    df_pred = pd.DataFrame({
        "model_name": model_name,
        "split": split_name,
        "patient_id": patient_ids,
        "volume_name": volume_names,
        "true_age": np.asarray(trues, dtype=float),
        pred_col: np.asarray(preds, dtype=float),
    })
    df_pred["brain_age_gap"] = df_pred[pred_col] - df_pred["true_age"]
    df_pred["abs_error"] = np.abs(df_pred["brain_age_gap"])
    return df_pred


# ============================================================
# Helper: 3D Grad-CAM Heatmaps
# ============================================================
"""
Finds the last Conv3d layer for Grad-CAM.

For ResNet and MedicalNet architectures, this is typically
a late convolutional layer within layer4.
"""
def find_last_conv3d(model):
    
    last_name, last_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            last_name, last_module = name, module
    return last_name, last_module

'''
Computes a simple 3D Grad-CAM for age regression.

The target quantity for the Grad-CAM computation is the model's
predicted age output itself.

The resulting activation map highlights the spatial regions that
contribute most strongly to the age prediction.
'''
def compute_gradcam_3d(model, input_tensor, target_layer):
  
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    output = model_forward_age(model, input_tensor)
    score = output[0]
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    if not activations or not gradients:
        return None

    act = activations[0]        # (1,C,d,h,w)
    grad = gradients[0]         # (1,C,d,h,w)
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="trilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()

    cam = cam - np.min(cam)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    return cam.astype(np.float32)

'''
Saves central slice overlays of the 3D heatmap.

Red and yellow regions indicate areas that contribute strongly
to the model's age prediction.

These visualizations help identify which anatomical structures
have the greatest influence on the predicted brain age.
'''
def save_gradcam_overlay(volume, cam, save_path, title="Grad-CAM"):

    volume = np.asarray(volume, dtype=np.float32)
    cam = np.asarray(cam, dtype=np.float32)
    z_indices = np.linspace(0, volume.shape[0] - 1, 6, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, z in zip(axes.ravel(), z_indices):
        ax.imshow(volume[z], cmap="gray")
        ax.imshow(cam[z], cmap="jet", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(f"Slice {z}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_heatmaps_for_model(model, dataset, model_name, max_items=3):
    layer_name, target_layer = find_last_conv3d(model)
    if target_layer is None:
        print("No Conv3d-layer for Grad-CAM found.")
        return []

    saved = []
    for idx in range(min(max_items, len(dataset))):
        xb, yb, wb, patient_id, volume_name = dataset[idx]
        input_tensor = xb.unsqueeze(0).to(device)
        cam = compute_gradcam_3d(model, input_tensor, target_layer)
        if cam is None:
            continue

        # xb has shape (1, D, H, W). Remove the channel dimension for visualization purposes so that the volume can be displayed directly as a 3D image stack.
        volume = xb[0].cpu().numpy()
        out_path = HEATMAP_DIR / f"{safe_filename(model_name)}_{safe_filename(volume_name)}_gradcam.png"
        save_gradcam_overlay(
            volume,
            cam,
            out_path,
            title=f"{model_name} Grad-CAM | patient {patient_id} | layer {layer_name}"
        )
        saved.append(str(out_path))
    return saved


# ============================================================
# Training + Evaluation Loop
# ============================================================
for model_name, (pth_file, block_type, layers) in model_files.items():
    if model_name not in RUN_MODEL_NAMES:
        continue

    print()
    print("=" * 40)
    print("Model:", model_name)
    print("=" * 40)

    batch_size = 4 if model_name in ["resnet10", "resnet18", "resnet34"] else batch_size_default
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_stack)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_stack)

    model = build_med3d_model(model_name, block_type, layers)

    # ---- Baseline Evaluation: Med3D-Architecture untrained ----
    # This baseline is on purpose BEFORE pretrained checkpoint and BEFORE training.
    baseline_trues, baseline_preds, baseline_ids, baseline_names = evaluate_model(model, val_loader, device, clip_range=PRED_CLIP_RANGE)
    baseline_core = compute_core_metrics(baseline_trues, baseline_preds)
    baseline_extra = compute_extra_metrics(baseline_trues, baseline_preds)
    baseline_metrics = {**baseline_core, **baseline_extra, "model_name": model_name, "baseline": "untrained_med3d"}
    baseline_results_all[model_name] = baseline_metrics

    baseline_pred_df = make_prediction_table(
        model_name,
        "val",
        baseline_trues,
        baseline_preds,
        baseline_ids,
        baseline_names,
        pred_col="pred_baseline_untrained"
    )
    baseline_pred_df.to_csv(SAVE_DIR / f"{model_name}_baseline_untrained_predictions.csv", index=False)
    save_metrics_csv(baseline_metrics, SAVE_DIR / f"{model_name}_baseline_untrained_metrics.csv")

    # ---- Load pretrained checkpoint, if applicable ----
    pretrained_loaded = load_pretrained_if_available(model, pth_file)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_mae = float("inf")
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0
    history = []

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        # Freeze batchNorm:  Prevents validation metrics from exploding when using very small batch sizes
        set_bn_eval(model)
        train_losses = []

        for xb, yb, wb, ids, names in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model_forward_age(model, xb)
            loss = weighted_mae_loss(pred, yb, wb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_trues, val_preds, val_ids_eval, val_names_eval = evaluate_model(model, val_loader, device, clip_range=PRED_CLIP_RANGE)
        core = compute_core_metrics(val_trues, val_preds)
        extra = compute_extra_metrics(val_trues, val_preds)
        ci_low, ci_high = bootstrap_ci_mae(val_trues, val_preds, n_bootstrap=500, random_state=SEED)

        val_mae = core["MAE"]
        val_rmse = extra["RMSE"]
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - start_time

        history_row = {
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_bag_mean": core["BAG_mean"],
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
        }
        history.append(history_row)

        print(
            f"[{model_name}] Epoch {epoch}/{EPOCHS} "
            f"TrainWeightedMAE={mean_train_loss:.4f} ValMAE={val_mae:.4f} "
            f"ValRMSE={val_rmse:.4f} LR={current_lr:.2e}"
        )

        scheduler.step(val_mae)

        # ---- Save best model after Validation-MAE ----
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save({
                "model_name": model_name,
                "epoch": epoch,
                "state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mae": best_val_mae,
                "pretrained_loaded": pretrained_loaded,
                "config": {
                    "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "batch_size": batch_size,
                    "softplus_output": True,
                    "weighted_mae_loss": True,
                    "train_augmentation": True,
                }
            }, SAVE_DIR / f"{model_name}_best_checkpoint.pt")
        else:
            epochs_without_improvement += 1

        # Last Checkpoint, allows long-running training runs to be resumed from a saved checkpoint
        torch.save({
            "model_name": model_name,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae": best_val_mae,
            "pretrained_loaded": pretrained_loaded,
        }, SAVE_DIR / f"{model_name}_last_checkpoint.pt")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}, best ValMAE={best_val_mae:.4f}")
            break

    # ---- Save history ----
    history_df = pd.DataFrame(history)
    history_df.to_csv(SAVE_DIR / f"{model_name}_training_history.csv", index=False)

    # ---- Load best model and final evaluation ----
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), SAVE_DIR / f"{model_name}_best_state_dict.pth")

    val_trues, val_preds, val_ids_eval, val_names_eval = evaluate_model(model, val_loader, device, clip_range=PRED_CLIP_RANGE)
    core = compute_core_metrics(val_trues, val_preds)
    extra = compute_extra_metrics(val_trues, val_preds)
    ci_low, ci_high = bootstrap_ci_mae(val_trues, val_preds, n_bootstrap=1000, random_state=SEED)

    pred_df = make_prediction_table(
        model_name,
        "val",
        val_trues,
        val_preds,
        val_ids_eval,
        val_names_eval,
        pred_col="pred_model_best"
    )
    pred_df = pred_df.merge(
        baseline_pred_df[["patient_id", "volume_name", "pred_baseline_untrained"]],
        on=["patient_id", "volume_name"],
        how="left"
    )
    pred_df["baseline_brain_age_gap"] = pred_df["pred_baseline_untrained"] - pred_df["true_age"]
    pred_df.to_csv(SAVE_DIR / f"{model_name}_predictions.csv", index=False)
    all_prediction_tables.append(pred_df)

    # ---- Collect and save metrics ----
    results_all[model_name] = {
        "model_name": model_name,
        "pretrained_loaded": pretrained_loaded,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "final_MAE": core["MAE"],
        "MAE_CI_low": ci_low,
        "MAE_CI_high": ci_high,
        "RMSE": extra["RMSE"],
        "MedianAE": extra["MedianAE"],
        "BAG_mean": core["BAG_mean"],
        "SD_BAG": extra["SD_BAG"],
        "MeanBias": extra["MeanBias"],
        "R2": core["R2"],
        "Spearman_rho": core["Spearman_rho"],
        "Spearman_p": core["Spearman_p"],
        "pct_within_3": extra["pct_within_3"],
        "pct_within_5": extra["pct_within_5"],
        "pct_within_10": extra["pct_within_10"],
        "baseline_untrained_MAE": baseline_core["MAE"],
        "baseline_untrained_RMSE": baseline_extra["RMSE"],
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "batch_size": batch_size,
        "weighted_loss": True,
        "softplus_output": True,
        "train_augmentation": True,
    }
    save_metrics_csv(results_all[model_name], SAVE_DIR / f"{model_name}_best_metrics.csv")

    # ---- Plots ----
    plot_training_curves(history_df, PLOT_DIR, model_name)
    plot_pred_vs_actual_dual(
        val_trues,
        baseline_pred_df["pred_baseline_untrained"].values,
        val_preds,
        PLOT_DIR,
        title=f"{model_name} Pred vs Actual Baseline vs Best Model"
    )
    plot_residuals_vs_actual_dual(
        val_trues,
        baseline_pred_df["pred_baseline_untrained"].values,
        val_preds,
        PLOT_DIR,
        title=f"{model_name} Residuals Baseline vs Best Model"
    )
    plot_bag_hist_dual(
        val_trues,
        baseline_pred_df["pred_baseline_untrained"].values,
        val_preds,
        PLOT_DIR,
        title=f"{model_name} BAG Histogram Baseline vs Best Model"
    )
    plot_bland_altman_dual(
        val_trues,
        baseline_pred_df["pred_baseline_untrained"].values,
        val_preds,
        PLOT_DIR,
        title=f"{model_name} Bland Altman Baseline vs Best Model"
    )

    # ---- Heatmaps ----
    try:
        saved_heatmaps = generate_heatmaps_for_model(model, val_dataset, model_name, max_items=N_HEATMAPS_PER_MODEL)
        pd.DataFrame({"model_name": model_name, "heatmap_path": saved_heatmaps}).to_csv(
            SAVE_DIR / f"{model_name}_heatmaps.csv",
            index=False
        )
    except Exception as e:
        print(f"Heatmap generation failed for {model_name}: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- Save results ----
df_results = pd.DataFrame(results_all).T
df_results.to_csv(SAVE_DIR / "med3d_results.csv", index=False)
print("Saved results CSV to", SAVE_DIR / "med3d_results.csv")

df_baseline = pd.DataFrame(baseline_results_all).T
df_baseline.to_csv(SAVE_DIR / "med3d_untrained_baseline_results.csv", index=False)
print("Saved baseline CSV to", SAVE_DIR / "med3d_untrained_baseline_results.csv")

if all_prediction_tables:
    pd.concat(all_prediction_tables, ignore_index=True).to_csv(SAVE_DIR / "med3d_all_predictions.csv", index=False)
    print("Saved all predictions CSV to", SAVE_DIR / "med3d_all_predictions.csv")




