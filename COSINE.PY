#!/usr/bin/env python3
"""
🎯 FINAL CONFORMAL PREDICTION EXPERIMENT
Comprehensive comparison of THR vs APS vs RAPS with full metrics and analysis.

This script implements the complete experimental pipeline with:
- Three methods: THR (threshold-based) vs APS (standard adaptive) vs RAPS (safe/regularized)
- Full K-shot range: [1, 2, 5, 10, 20, 50]
- Complete trial counts for robust statistics
- Advanced rank-based metrics (MRTL, R1CR, MRR, RTC)
- Clinical workload metric (RTC: Rank to Correct)
- Final analysis and visualization
"""

# =============================================================================
# SETUP AND IMPORTS
# =============================================================================

try:
    import google.colab
    IN_COLAB = True
    print("🔧 Running in Google Colab")
    
    import subprocess
    import sys
    
    # Install required packages
    packages = ['scikit-learn', 'matplotlib', 'seaborn']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
except ImportError:
    IN_COLAB = False
    print("🖥️ Running locally")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Experimental Configuration
K_SHOT_VALUES = [1, 2, 5, 10, 20, 50]
TRIAL_COUNTS = {1: 50, 2: 50, 5: 40, 10: 30, 20: 25, 50: 20}  # More trials for lower K
VAL_SAMPLES_PER_CLASS = 20
CALIB_MIN_PER_CLASS = 10
CALIB_MAX_PER_CLASS = 30

# Multiple coverage levels to test
COVERAGE_LEVELS = [0.90, 0.95, 0.99]  # 90%, 95%, 99% coverage
ALPHA_VALUES = [0.10, 0.05, 0.01]     # Corresponding alpha values

# Default for backward compatibility
ALPHA = 0.1  # 90% target coverage (default)
TARGET_COVERAGE = 1 - ALPHA

CLASSIFIER_NAME = "Cosine classifier"

print(f"🚀 Using device: {DEVICE}")
print(f"🎯 Target coverage: {TARGET_COVERAGE:.1%}")

# =============================================================================
# COSINE CLASSIFIER
# =============================================================================

class CosineClassifier(nn.Module):
    """Cosine classifier with temperature scaling"""

    def __init__(self, feature_dim, num_classes, temperature: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.W = nn.Parameter(torch.randn(feature_dim, num_classes, device=DEVICE))
        nn.init.xavier_uniform_(self.W)
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        # Add classes_ attribute for compatibility with conformal prediction functions
        self.classes_ = np.arange(num_classes)

    def forward(self, x, temperature: float = None):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        w_norm = torch.nn.functional.normalize(self.W, p=2, dim=0)
        cosine_sim = torch.matmul(x_norm, w_norm)
        T = self.temperature if temperature is None else torch.tensor(float(temperature), device=x.device)
        T = torch.clamp(T, min=1e-6)
        logits = cosine_sim / T
        return torch.softmax(logits, dim=1)

    def predict_proba(self, X, temperature: float = None):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            probs = self.forward(X_tensor, temperature=temperature)
            return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def fit(self, X_support, y_support, X_val=None, y_val=None, epochs=200, lr=0.01, batch_size=32, augment=False):
        self.train()
        X_tensor = torch.tensor(X_support, dtype=torch.float32, device=DEVICE)
        y_tensor = torch.tensor(y_support, dtype=torch.long, device=DEVICE)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_loss = float('inf')
        best_state = None

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= max(len(dataloader), 1)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.clone().detach() for k, v in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict(best_state)

        if X_val is not None and len(X_val) > 0:
            self.tune_temperature(X_val, y_val)

    def tune_temperature(self, X_val, y_val, temps=None):
        if temps is None:
            temps = np.logspace(-1, 1, num=21)
        best_T = float(self.temperature.item())
        best_loss = float('inf')
        X_tensor = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        y_tensor = torch.tensor(y_val, dtype=torch.long, device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        for T in temps:
            with torch.no_grad():
                probs = self.forward(X_tensor, temperature=T)
                loss = criterion(probs, y_tensor).item()
            if loss < best_loss:
                best_loss = loss
                best_T = T
        self.temperature.data = torch.tensor(best_T, device=DEVICE)
        return self


# =============================================================================
# CONFORMAL PREDICTION METHODS
# =============================================================================

class THR():
    """
    Threshold-based conformal prediction method.
    Uses probability of true class as conformity score.
    """
    def __init__(self, softmax, true_class, alpha):
        self.prob_output = softmax
        self.true_class = true_class
        # Remove incorrect finite-sample correction that made sets smaller
        self.alpha = alpha
        

    def conformal_score(self):
        conformal_score = self.prob_output[range(self.prob_output.shape[0]), self.true_class]
        return conformal_score
    

    def quantile(self):
        conformal_scores = self.conformal_score()
        # Conservative quantile for finite-sample safety
        quantile_value = torch.quantile(conformal_scores, self.alpha, interpolation='higher')
        return quantile_value


    def prediction(self, softmax, quantile_value):
        prob_output = softmax
        predictions = prob_output >= quantile_value
        predictions = predictions.int()
        return predictions


def thr_conformal_prediction(model, calib_X, calib_y, test_X, alpha=0.1):
    """THR conformal prediction wrapper - FIXED"""
    # Get probabilities
    calib_probs = torch.tensor(model.predict_proba(calib_X), dtype=torch.float32, device=DEVICE)
    test_probs = torch.tensor(model.predict_proba(test_X), dtype=torch.float32, device=DEVICE)
    
    # Map labels to classifier's class indices
    class_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
    
    # Filter calibration data to only include classes seen during training
    calib_mask = np.array([label in class_to_idx for label in calib_y])
    calib_X_filtered = calib_X[calib_mask]
    calib_y_filtered = calib_y[calib_mask]
    calib_probs_filtered = calib_probs[calib_mask]
    
    # Map labels to classifier indices
    calib_labels_mapped = np.array([class_to_idx[label] for label in calib_y_filtered])
    calib_labels = torch.tensor(calib_labels_mapped, dtype=torch.long, device=DEVICE)
    
    # Conformity scores: p_true
    calib_scores = calib_probs_filtered[torch.arange(len(calib_labels), device=DEVICE), calib_labels]
    quantile_value = torch.quantile(calib_scores, alpha, interpolation='higher')
    predictions = (test_probs >= quantile_value).int()
    
    # Convert to prediction sets - ROBUST CONVERSION
    prediction_sets = []
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    for i in range(len(test_X)):
        pred_set_mask = predictions[i]
        pred_indices = torch.nonzero(pred_set_mask, as_tuple=False).flatten()
        
        if len(pred_indices) == 0:
            pred_class_idx = model.predict([test_X[i]])[0]
            pred_set = [idx_to_class[pred_class_idx]]
        else:
            pred_set = [idx_to_class[idx.item()] for idx in pred_indices]
            
        prediction_sets.append(pred_set)
    
    return prediction_sets, test_probs.cpu().numpy()


class APS_Custom():
    """Adaptive Prediction Sets - standard adaptive baseline"""
    def __init__(self, softmax, true_class, alpha):
        self.prob_output = softmax
        self.true_class = true_class
        # Use exact (1 - alpha) level (no finite-sample inflation)
        self.q_level = max(0.0, min(1.0, (1 - alpha)))
        
    def conformal_score(self):
        conformal_score = []
        for i in range(self.prob_output.shape[0]):
            true_class_prob = self.prob_output[i][self.true_class[i]]
            current_class_prob = self.prob_output[i]
            sorted_class_prob, _ = torch.sort(current_class_prob, descending=True)
            index = torch.nonzero(sorted_class_prob == true_class_prob).item()
            cumulative_sum = torch.sum(sorted_class_prob[:index + 1])
            conformal_score.append(cumulative_sum)
        conformal_score = torch.tensor(conformal_score)
        
        return conformal_score

    def quantile(self):
        conformal_scores = self.conformal_score()
        # Less conservative interpolation to reduce overcoverage
        quantile_value = torch.quantile(conformal_scores, self.q_level, interpolation='lower')
        return quantile_value
    
    def prediction(self, softmax, quantile_value):
        # Include the minimal k such that cumulative mass >= threshold
        sorted_probs, sorted_idx = torch.sort(softmax, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        prediction = torch.zeros_like(softmax)
        for i in range(softmax.shape[0]):
            k_include = int((cumsum[i] < quantile_value).sum().item()) + 1
            prediction[i, sorted_idx[i, :k_include]] = 1.0
        return prediction


class RAPS():
    """Regularized Adaptive Prediction Sets - safe/conservative method"""
    def __init__(self, softmax, true_class, alpha, k_reg=5, lambd=0.01, rand=True):
        self.prob_output = softmax
        self.true_class = true_class
        # Use exact (1 - alpha) level (no finite-sample inflation)
        self.q_level = max(0.0, min(1.0, (1 - alpha)))
        self.k_reg = k_reg
        self.lambd = lambd
        self.rand = rand

    def conformal_score(self):
        # RAPS conformity score: cumulative mass to true rank plus regularization
        sorted_probs, indices = torch.sort(self.prob_output, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        true_pos = (indices == self.true_class.view(-1, 1)).nonzero(as_tuple=False)[:, 1]
        reg = torch.clamp(true_pos - self.k_reg, min=0).to(torch.float32) * self.lambd
        base = cumsum[torch.arange(self.prob_output.shape[0]), true_pos]
        score = base + reg
        if self.rand:
            jitter = torch.rand(self.prob_output.shape[0]) * sorted_probs[torch.arange(self.prob_output.shape[0]), true_pos]
            score = score - jitter
        return score

    def quantile(self):
        conformal_scores = self.conformal_score()
        # Less conservative interpolation to reduce overcoverage
        quantile_value = torch.quantile(conformal_scores, self.q_level, interpolation='lower')
        return quantile_value
    
    def prediction(self, softmax, quantile_value):
        # Include the minimal k such that (cumulative + regularization) >= threshold
        sorted_probs, sorted_idx = torch.sort(softmax, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=1)
        n_classes = softmax.shape[1]
        idx = torch.arange(n_classes, dtype=torch.float32)
        penalty = torch.clamp(idx - (self.k_reg - 1), min=0) * self.lambd
        prediction = torch.zeros_like(softmax)
        for i in range(softmax.shape[0]):
            cumsum_reg = cumsum[i] + penalty
            k_include = int((cumsum_reg < quantile_value).sum().item()) + 1
            prediction[i, sorted_idx[i, :k_include]] = 1.0
        return prediction


def aps_conformal_prediction(model, calib_X, calib_y, test_X, alpha=0.1):
    """APS conformal prediction wrapper"""
    # Get probabilities
    calib_probs = torch.tensor(model.predict_proba(calib_X), dtype=torch.float32, device=DEVICE)
    test_probs = torch.tensor(model.predict_proba(test_X), dtype=torch.float32, device=DEVICE)
    
    # Map labels to classifier's class indices
    class_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
    
    # Filter calibration data to only include classes seen during training
    calib_mask = np.array([label in class_to_idx for label in calib_y])
    calib_X_filtered = calib_X[calib_mask]
    calib_y_filtered = calib_y[calib_mask]
    calib_probs_filtered = calib_probs[calib_mask]
    
    # Map labels to classifier indices
    calib_labels_mapped = np.array([class_to_idx[label] for label in calib_y_filtered])
    calib_labels = torch.tensor(calib_labels_mapped, dtype=torch.long, device=DEVICE)
    
    # Apply APS
    aps = APS_Custom(calib_probs_filtered, calib_labels, alpha)
    quantile_value = aps.quantile()
    predictions = aps.prediction(test_probs, quantile_value)
    
    # Convert to prediction sets
    prediction_sets = []
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    for i in range(len(test_X)):
        pred_set_mask = predictions[i]
        pred_indices = torch.nonzero(pred_set_mask, as_tuple=False).flatten()
        
        if len(pred_indices) == 0:
            pred_class_idx = model.predict([test_X[i]])[0]
            pred_set = [idx_to_class[pred_class_idx]]
        else:
            pred_set = [idx_to_class[idx.item()] for idx in pred_indices]
            
        prediction_sets.append(pred_set)
    
    return prediction_sets, test_probs.cpu().numpy()


def raps_conformal_prediction(model, calib_X, calib_y, test_X, alpha=0.1, k_reg=5, lambd=0.01):
    """RAPS conformal prediction wrapper"""
    # Get probabilities
    calib_probs = torch.tensor(model.predict_proba(calib_X), dtype=torch.float32, device=DEVICE)
    test_probs = torch.tensor(model.predict_proba(test_X), dtype=torch.float32, device=DEVICE)
    
    # Map labels to classifier's class indices
    class_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
    
    # Filter calibration data to only include classes seen during training
    calib_mask = np.array([label in class_to_idx for label in calib_y])
    calib_X_filtered = calib_X[calib_mask]
    calib_y_filtered = calib_y[calib_mask]
    calib_probs_filtered = calib_probs[calib_mask]
    
    # Map labels to classifier indices
    calib_labels_mapped = np.array([class_to_idx[label] for label in calib_y_filtered])
    calib_labels = torch.tensor(calib_labels_mapped, dtype=torch.long, device=DEVICE)
    
    # Apply RAPS
    raps = RAPS(calib_probs_filtered, calib_labels, alpha, k_reg=k_reg, lambd=lambd, rand=True)
    quantile_value = raps.quantile()
    predictions = raps.prediction(test_probs, quantile_value)
    
    # Convert to prediction sets
    prediction_sets = []
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    for i in range(len(test_X)):
        pred_set_mask = predictions[i]
        pred_indices = torch.nonzero(pred_set_mask, as_tuple=False).flatten()
        
        if len(pred_indices) == 0:
            pred_class_idx = model.predict([test_X[i]])[0]
            pred_set = [idx_to_class[pred_class_idx]]
        else:
            pred_set = [idx_to_class[idx.item()] for idx in pred_indices]
            
        prediction_sets.append(pred_set)
    
    return prediction_sets, test_probs.cpu().numpy()


# =============================================================================
# RANK-BASED METRICS
# =============================================================================

def calculate_rank_metrics(prediction_sets, true_labels, probabilities, penalize_miss: bool = True):
    """
    Calculate advanced rank-based metrics using ACTUAL prediction set order:
    - MRTL: Mean Rank of True Label within prediction set
    - R1CR: Rank-1 Containment Rate (is true label first in prediction set?)
    - MRR: Mean Reciprocal Rank within prediction set
    - RTC: Rank to Correct (clinical workload metric)
    """
    mrtl_scores = []
    r1cr_scores = []
    mrr_scores = []
    rtc_scores = []
    
    for i, (pred_set, true_label, probs) in enumerate(zip(prediction_sets, true_labels, probabilities)):
        # Check if true label is in prediction set
        if int(true_label) not in pred_set:
            if penalize_miss:
                true_rank = len(pred_set) + 1
                mrtl_scores.append(true_rank)
            r1cr_scores.append(0)
            mrr_scores.append(0.0)
            rtc_scores.append(max(0, len(pred_set) - 1))
            continue
        
        # Find position of true label within the ACTUAL prediction set order
        try:
            true_rank = pred_set.index(int(true_label)) + 1  # 1-indexed rank
        except ValueError:
            # Fallback if somehow not found
            true_rank = len(pred_set) + 1
            mrtl_scores.append(true_rank)
            r1cr_scores.append(0)
            mrr_scores.append(0.0)
            rtc_scores.append(true_rank - 1)
            continue
        
        # MRTL: Mean Rank of True Label within prediction set
        mrtl_scores.append(true_rank)
        
        # R1CR: Rank-1 Containment Rate (is true label first in prediction set?)
        r1cr_scores.append(1 if true_rank == 1 else 0)
        
        # MRR: Mean Reciprocal Rank within prediction set
        mrr_scores.append(1.0 / true_rank)
        
        # RTC: Rank to Correct (clinical workload metric)
        # Number of incorrect predictions clinician must review before finding correct one
        rtc_scores.append(true_rank - 1)
    
    return {
        'mrtl': np.mean(mrtl_scores) if len(mrtl_scores) > 0 else 0.0,
        'r1cr': np.mean(r1cr_scores),
        'mrr': np.mean(mrr_scores),
        'rtc': np.mean(rtc_scores),
        'mrtl_std': np.std(mrtl_scores) if len(mrtl_scores) > 0 else 0.0,
        'r1cr_std': np.std(r1cr_scores),
        'mrr_std': np.std(mrr_scores),
        'rtc_std': np.std(rtc_scores)
    }


def calculate_conformal_metrics(prediction_sets, true_labels, alpha):
    """Calculate standard conformal prediction metrics"""
    coverage_list = []
    set_sizes = []
    empty_count = 0
    
    for pred_set, true_label in zip(prediction_sets, true_labels):
        if len(pred_set) == 0:
            empty_count += 1
        
        is_covered = int(true_label) in pred_set
        coverage_list.append(is_covered)
        set_sizes.append(len(pred_set))
    
    coverage = np.mean(coverage_list)
    avg_set_size = np.mean(set_sizes)
    empty_rate = empty_count / len(true_labels)
    
    # ESR: fraction of ALL samples that are correct singletons
    # Also report helpful components: singleton_rate and singleton_accuracy
    n_samples = len(true_labels)
    is_singleton = np.array(set_sizes) == 1
    num_singletons = int(is_singleton.sum())
    num_correct_singletons = int(sum(
        int(int(true_labels[i]) in prediction_sets[i]) for i in range(n_samples) if is_singleton[i]
    ))
    esr = (num_correct_singletons / n_samples) if n_samples > 0 else 0.0
    singleton_rate = (num_singletons / n_samples) if n_samples > 0 else 0.0
    singleton_accuracy = (num_correct_singletons / num_singletons) if num_singletons > 0 else 0.0

    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'empty_rate': empty_rate,
        'coverage_gap': abs(coverage - (1 - alpha)),
        'coverage_std': np.std(coverage_list),
        'set_size_std': np.std(set_sizes),
        'esr': esr,
        'singleton_rate': singleton_rate,
        'singleton_accuracy': singleton_accuracy
    }


# =============================================================================
# DATA LOADING AND EXPERIMENT SETUP
# =============================================================================

def load_embeddings(embedding_file: str = None):
    """Load embeddings with optional explicit path (supports Colab /content).

    If embedding_file is provided, load that; otherwise try common defaults.
    """
    possible_paths = [
        # Colab examples
        Path("/content/cric_uni_embeddings_complete.pt"),
        Path("/content/cric_cell_embeddings_complete.pt"),
        Path("/content/ebhi_clip_embeddings.pt"),
        # Kaggle common locations
        Path("/kaggle/working/gastric_embeddings.pt"),
        Path("/kaggle/working/gastric_clip_embeddings.pt"),
        Path("/kaggle/working/gastric_cancer_embeddings.pt"),
        Path("/kaggle/input/gastric_embeddings.pt"),
        Path("/kaggle/input/gastric_clip_embeddings.pt"),
        Path("/kaggle/input/gastric_cancer_embeddings.pt"),
        # Local workspace fallbacks
        Path("gastric_embeddings.pt"),
        Path("gastric_clip_embeddings.pt"),
        Path("gastric_cancer_embeddings.pt"),
        Path("cric_uni_embeddings_complete.pt"),
        Path("cric_cell_embeddings_complete.pt"),
        Path("ebhi_clip_embeddings.pt"),
    ]

    embeddings_path = None
    if embedding_file is not None:
        p = Path(embedding_file)
        if p.exists():
            embeddings_path = p
        else:
            print(f"❌ Provided embeddings path not found: {embedding_file}")
    if embeddings_path is None:
        for path in possible_paths:
            if path.exists():
                embeddings_path = path
                break
    
    if embeddings_path is None:
        print("❌ Embeddings file not found!")
        if IN_COLAB:
            print("💡 Upload file using: upload_embeddings_colab()")
        return None, None, None
    
    try:
        data = torch.load(embeddings_path, weights_only=False, map_location=DEVICE)

        # Robust key handling for different embedding dump formats
        embeddings = data.get('embeddings')
        if embeddings is None:
            embeddings = data.get('features')
        if embeddings is None:
            embeddings = data.get('X')
        if embeddings is None:
            embeddings = data.get('x')
            
        labels = data.get('labels')
        if labels is None:
            labels = data.get('y')
        if labels is None:
            labels = data.get('targets')
        if labels is None:
            labels = data.get('target')

        # Class mapping resolution (prefer id->name)
        class_map = None
        if 'class_map' in data:
            class_map = data['class_map']
        elif 'class_mapping' in data:
            class_map = data['class_mapping']
        elif 'idx_to_class' in data:
            class_map = data['idx_to_class']
        elif 'class_to_idx' in data and isinstance(data['class_to_idx'], dict):
            class_map = {int(v): k for k, v in data['class_to_idx'].items()}
        elif 'classes' in data and isinstance(data['classes'], (list, tuple)):
            class_map = {int(i): str(name) for i, name in enumerate(data['classes'])}

        if embeddings is None or labels is None:
            raise ValueError("Embeddings or labels not found in the loaded file. Expected keys like 'embeddings'/'features' and 'labels'/'y'.")

        # Convert to numpy arrays if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Ensure labels are a 1D integer numpy array
        labels = np.asarray(labels)
        if labels.ndim > 1:
            labels = labels.squeeze()

        # If labels are strings, map to integer ids consistently
        if labels.dtype.kind in {'U', 'S', 'O'}:
            unique_names = sorted({str(v) for v in labels})
            name_to_id = {name: i for i, name in enumerate(unique_names)}
            labels = np.array([name_to_id[str(v)] for v in labels], dtype=np.int64)
            if class_map is None:
                class_map = {i: name for name, i in name_to_id.items()}

        # Final guard: derive class_map if missing
        if class_map is None:
            unique_ids = sorted(set(int(v) for v in labels.tolist()))
            class_map = {cid: str(cid) for cid in unique_ids}

        print(f"✅ Loaded {len(embeddings)} embeddings from {embeddings_path}")
        print(f"Classes: {class_map}")
        try:
            print(f"Embedding shape: {embeddings.shape}")
        except Exception:
            pass

        return embeddings, labels, class_map
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return None, None, None


def upload_embeddings_colab():
    """Upload embeddings in Colab"""
    if not IN_COLAB:
        print("❌ This function is only for Google Colab")
        return False
    
    try:
        from google.colab import files
        print("📁 Please select your 'ebhi_clip_embeddings.pt' file:")
        uploaded = files.upload()
        
        if 'ebhi_clip_embeddings.pt' in uploaded:
            print("✅ Embeddings file uploaded successfully!")
            return True
        else:
            print("❌ Please upload a file named 'ebhi_clip_embeddings.pt'")
            return False
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def create_data_splits(embeddings, labels, random_state=42):
    """Create pools for support/train, calibration, and test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        embeddings, labels, test_size=0.4, random_state=random_state, stratify=labels
    )
    X_calib_pool, X_test_pool, y_calib_pool, y_test_pool = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    print(f"📊 Pools:")
    print(f"  Support/Train pool: {len(X_train)}")
    print(f"  Calibration pool:   {len(X_calib_pool)}")
    print(f"  Test/Query pool:    {len(X_test_pool)}")

    return (X_train, y_train), (X_calib_pool, y_calib_pool), (X_test_pool, y_test_pool)


def sample_k_shot_data(X_train, y_train, k_shot, random_state=None):
    rng = np.random.default_rng(random_state)
    shot_indices = []
    for class_id in np.unique(y_train):
        class_indices = np.where(y_train == class_id)[0]
        take = min(k_shot, len(class_indices))
        if take == 0:
            continue
        selected = rng.choice(class_indices, size=take, replace=False)
        shot_indices.extend(selected)
    shot_indices = np.array(shot_indices, dtype=int)
    return X_train[shot_indices], y_train[shot_indices], shot_indices


def sample_validation_data(X_pool, y_pool, per_class=VAL_SAMPLES_PER_CLASS, random_state=None):
    rng = np.random.default_rng(random_state)
    val_indices = []
    for class_id in np.unique(y_pool):
        class_indices = np.where(y_pool == class_id)[0]
        if len(class_indices) == 0:
            continue
        take = min(per_class, len(class_indices))
        selected = rng.choice(class_indices, size=take, replace=False)
        val_indices.extend(selected)
    val_indices = np.array(val_indices, dtype=int)
    mask = np.ones(len(y_pool), dtype=bool)
    mask[val_indices] = False
    X_remaining, y_remaining = X_pool[mask], y_pool[mask]
    X_val, y_val = X_pool[val_indices], y_pool[val_indices]
    return (X_remaining, y_remaining), (X_val, y_val)


def sample_calibration_data(X_calib_pool, y_calib_pool, min_per_class=CALIB_MIN_PER_CLASS, max_per_class=CALIB_MAX_PER_CLASS, random_state=None):
    rng = np.random.default_rng(random_state)
    idx = []
    for class_id in np.unique(y_calib_pool):
        class_indices = np.where(y_calib_pool == class_id)[0]
        cls_n = len(class_indices)
        if cls_n == 0:
            continue
        desired = max(min_per_class, min(max_per_class, cls_n))
        take = min(desired, cls_n)
        selected = rng.choice(class_indices, size=take, replace=False)
        idx.extend(selected)
    idx = np.array(idx, dtype=int)
    return X_calib_pool[idx], y_calib_pool[idx]


# =============================================================================
# MAIN EXPERIMENT FUNCTIONS
# =============================================================================

def run_single_trial(k_shot, trial_seed, train_data, calib_data, test_data, method='thr', alpha=None):
    """Run a single trial with specified method"""
    X_train_pool, y_train_pool = train_data
    X_calib_pool, y_calib_pool = calib_data
    X_test, y_test = test_data
    
    try:
        # Sample K-shot support data
        X_support, y_support, support_idx = sample_k_shot_data(X_train_pool, y_train_pool, k_shot, random_state=trial_seed)

        # Build a validation set from the remaining pool (excluding support indices)
        mask = np.ones(len(y_train_pool), dtype=bool)
        mask[support_idx] = False
        X_remaining, y_remaining = X_train_pool[mask], y_train_pool[mask]
        (X_remaining, y_remaining), (X_val, y_val) = sample_validation_data(X_remaining, y_remaining, per_class=VAL_SAMPLES_PER_CLASS, random_state=trial_seed + 4242)

        # Get number of classes and feature dimension
        num_classes = len(np.unique(y_train_pool))
        feature_dim = X_support.shape[1]

        # Build class index mapping based on actual dataset classes
        all_classes = np.unique(y_train_pool)
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        
        # Remap labels to contiguous indices [0..C-1] for training/validation compatibility
        y_support_mapped = np.array([class_to_idx[int(lbl)] for lbl in y_support], dtype=np.int64)
        y_val_mapped = np.array([class_to_idx[int(lbl)] for lbl in y_val], dtype=np.int64) if (X_val is not None and y_val is not None and len(y_val) > 0) else y_val

        # Train cosine classifier on support set with validation tuning
        clf = CosineClassifier(feature_dim=feature_dim, num_classes=num_classes)
        # Expose classes_ consistent with column order
        clf.classes_ = all_classes
        clf.fit(X_support, y_support_mapped, X_val=X_val, y_val=y_val_mapped, epochs=200, lr=0.01, batch_size=32, augment=True)

        # Sample calibration data per class (10-30)
        X_calib, y_calib = sample_calibration_data(X_calib_pool, y_calib_pool, min_per_class=CALIB_MIN_PER_CLASS, max_per_class=CALIB_MAX_PER_CLASS, random_state=trial_seed)

        # Use provided alpha or default
        alpha_to_use = alpha if alpha is not None else ALPHA
        
        # Apply conformal prediction
        if method == 'thr':
            prediction_sets, probabilities = thr_conformal_prediction(
                clf, X_calib, y_calib, X_test, alpha=alpha_to_use
            )
        elif method == 'aps':
            prediction_sets, probabilities = aps_conformal_prediction(
                clf, X_calib, y_calib, X_test, alpha=alpha_to_use
            )
        elif method == 'raps':
            prediction_sets, probabilities = raps_conformal_prediction(
                clf, X_calib, y_calib, X_test, alpha=alpha_to_use, k_reg=5, lambd=0.01
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate all metrics
        conformal_metrics = calculate_conformal_metrics(prediction_sets, y_test, alpha_to_use)
        rank_metrics = calculate_rank_metrics(prediction_sets, y_test, probabilities, penalize_miss=True)
        
        # Combine results
        results = {
            'k_shot': k_shot,
            'trial_seed': trial_seed,
            'method': method,
            'accuracy': rank_metrics['r1cr'],  # R1CR is accuracy
            **conformal_metrics,
            **rank_metrics
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Trial failed (K={k_shot}, seed={trial_seed}, method={method}): {e}")
        return None


def run_method_comparison_experiment(embedding_file: str = None):
    """
    Step 2: Run the full-scale experiment for Simple_CP vs APS vs RAPS
    """
    print("🚀 FINAL CONFORMAL PREDICTION EXPERIMENT")
    print("="*70)
    print(f"🧠 Base classifier: {CLASSIFIER_NAME} (learned weights)")
    print("📊 Comparing THREE methods:")
    print("   🔹 THR (threshold-based baseline)")
    print("   🔹 APS (standard adaptive baseline)")  
    print("   🔹 RAPS (safe/regularized method)")
    print(f"🎯 K-shot values: {K_SHOT_VALUES}")
    print(f"🎯 Trial counts: {TRIAL_COUNTS}")
    print()
    
    # Load data
    embeddings, labels, class_map = load_embeddings(embedding_file)
    if embeddings is None:
        return None
    
    # Create splits
    train_data, calib_data, test_data = create_data_splits(embeddings, labels)
    
    # Methods to compare
    methods = ['thr', 'aps', 'raps']
    method_names = {'thr': 'THR', 'aps': 'APS', 'raps': 'RAPS'}
    
    all_results = {}
    
    # Loop through each coverage level
    for coverage_idx, (target_coverage, alpha) in enumerate(zip(COVERAGE_LEVELS, ALPHA_VALUES)):
        print(f"\n🎯 COVERAGE LEVEL {coverage_idx + 1}/{len(COVERAGE_LEVELS)}: {target_coverage:.0%} (α={alpha})")
        print("="*60)
        
        all_results[f'coverage_{target_coverage:.0%}'] = {}
        
        for method in methods:
            print(f"\n🔬 Running experiments for {method_names[method]} at {target_coverage:.0%} coverage...")
            all_results[f'coverage_{target_coverage:.0%}'][method] = {}
            
            for k_shot in K_SHOT_VALUES:
                num_trials = TRIAL_COUNTS[k_shot]
                print(f"\n📊 K={k_shot}-shot, {num_trials} trials...")
                
                trial_results = []
                
                for trial in tqdm(range(num_trials), desc=f"{method_names[method]} K={k_shot} ({target_coverage:.0%})"):
                    result = run_single_trial(
                        k_shot, trial, train_data, calib_data, test_data, method=method, alpha=alpha
                    )
                    if result is not None:
                        trial_results.append(result)
                
                all_results[f'coverage_{target_coverage:.0%}'][method][k_shot] = trial_results
                
                # Quick summary for each K-shot
                if trial_results:
                    metrics = ['accuracy', 'coverage', 'avg_set_size', 'mrtl', 'mrr', 'rtc', 'esr']
                    summary = {}
                    for metric in metrics:
                        values = [r[metric] for r in trial_results]
                        summary[metric] = {'mean': np.mean(values), 'std': np.std(values)}
                    
                    print(f"  📈 {method_names[method]} K={k_shot} Summary:")
                    print(f"    Accuracy: {summary['accuracy']['mean']:.3f}±{summary['accuracy']['std']:.3f}")
                    print(f"    Coverage: {summary['coverage']['mean']:.3f}±{summary['coverage']['std']:.3f}")
                    print(f"    Set Size: {summary['avg_set_size']['mean']:.1f}±{summary['avg_set_size']['std']:.1f}")
                    print(f"    MRTL: {summary['mrtl']['mean']:.1f}±{summary['mrtl']['std']:.1f}")
                    print(f"    MRR: {summary['mrr']['mean']:.3f}±{summary['mrr']['std']:.3f}")
                    print(f"    RTC: {summary['rtc']['mean']:.1f}±{summary['rtc']['std']:.1f}")
                    print(f"    ESR: {summary['esr']['mean']:.3f}±{summary['esr']['std']:.3f}")
                    print()
    
    return all_results


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def generate_summary_table(all_results):
    """Step 4: Generate final summary table for all coverage levels"""
    print("\n📊 MULTI-COVERAGE SUMMARY TABLES")
    print("="*120)
    print(f"🧠 Classifier backbone: {CLASSIFIER_NAME}")
    
    methods = ['thr', 'aps', 'raps']
    method_names = {'thr': 'THR', 'aps': 'APS', 'raps': 'RAPS'}
    
    summary_data = {}
    
    # Loop through each coverage level
    for coverage_level in COVERAGE_LEVELS:
        coverage_key = f'coverage_{coverage_level:.0%}'
        if coverage_key not in all_results:
            continue
            
        print(f"\n🎯 COVERAGE LEVEL: {coverage_level:.0%}")
        print("="*80)
        
        # Header
        header = f"{'Method':<10} | {'K-Shot':<6} | {'Accuracy':<12} | {'Coverage':<12} | {'Set Size':<12} | {'MRTL':<12} | {'R1CR':<12} | {'MRR':<12} | {'RTC':<12} | {'ESR':<12}"
        print(header)
        print("-" * len(header))
        
        summary_data[coverage_key] = {}
        
        for method in methods:
            summary_data[coverage_key][method] = {}
            for k_shot in K_SHOT_VALUES:
                if k_shot in all_results[coverage_key][method] and all_results[coverage_key][method][k_shot]:
                    results = all_results[coverage_key][method][k_shot]
                    
                    # Calculate means and stds
                    metrics = {}
                    for metric in ['accuracy', 'coverage', 'avg_set_size', 'mrtl', 'r1cr', 'mrr', 'rtc', 'esr']:
                        values = [r[metric] for r in results]
                        metrics[metric] = {'mean': np.mean(values), 'std': np.std(values)}
                    
                    summary_data[coverage_key][method][k_shot] = metrics
                    
                    # Print row
                    row = f"{method_names[method]:<10} | {k_shot:<6} | "
                    row += f"{metrics['accuracy']['mean']:.3f}±{metrics['accuracy']['std']:.3f} | "
                    row += f"{metrics['coverage']['mean']:.3f}±{metrics['coverage']['std']:.3f} | "
                    row += f"{metrics['avg_set_size']['mean']:.1f}±{metrics['avg_set_size']['std']:.1f}   | "
                    row += f"{metrics['mrtl']['mean']:.1f}±{metrics['mrtl']['std']:.1f}   | "
                    row += f"{metrics['r1cr']['mean']:.3f}±{metrics['r1cr']['std']:.3f} | "
                    row += f"{metrics['mrr']['mean']:.3f}±{metrics['mrr']['std']:.3f} | "
                    row += f"{metrics['rtc']['mean']:.1f}±{metrics['rtc']['std']:.1f} | "
                    row += f"{metrics['esr']['mean']:.3f}±{metrics['esr']['std']:.3f}"
                    print(row)
    
    return summary_data


def create_visualizations(all_results):
    """Visualizations disabled as per request."""
    return None


def save_results(all_results, summary_data, embedding_file: str = None):
    """Save results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(embedding_file).stem if embedding_file else "embeddings"
    title = f"{base} + {CLASSIFIER_NAME}"
    safe_title = title.replace('/', '-').replace('\\', '-')
    filename = f"final_conformal {safe_title} {timestamp}.json"
    
    # Prepare data for JSON serialization
    results_to_save = {
        'experiment_info': {
            'timestamp': timestamp,
            'k_shot_values': K_SHOT_VALUES,
            'trial_counts': TRIAL_COUNTS,
            'alpha': ALPHA,
            'target_coverage': TARGET_COVERAGE,
            'methods_compared': ['THR', 'APS', 'RAPS']
        },
        'raw_results': all_results,
        'summary_statistics': summary_data
    }
    
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {filename}")
    return filename


def save_summary_csv(summary_data, embedding_file: str = None):
    """Save flattened summary_data to a CSV with the same metrics shown in the summary tables"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(embedding_file).stem if embedding_file else "embeddings"
    title = f"{base} + {CLASSIFIER_NAME}"
    safe_title = title.replace('/', '-').replace('\\', '-')
    csv_filename = f"final_conformal {safe_title} {timestamp}.csv"

    header = [
        'coverage', 'method', 'k_shot',
        'accuracy', 'coverage', 'avg_set_size', 'mrtl', 'r1cr', 'mrr', 'rtc', 'esr',
    ]

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for coverage_key, methods_dict in summary_data.items():
            if not isinstance(methods_dict, dict):
                continue
            coverage_label = str(coverage_key).replace('coverage_', '')
            for method, kdict in methods_dict.items():
                if not isinstance(kdict, dict):
                    continue
                for k_shot, metrics in kdict.items():
                    def combo(metric):
                        m = metrics.get(metric, {}).get('mean', np.nan)
                        s = metrics.get(metric, {}).get('std', np.nan)
                        if np.isnan(m) or np.isnan(s):
                            return ''
                        return f"{m:.3f} (mean) ± {s:.3f} (std dev)"

                    row = [
                        coverage_label,
                        method,
                        k_shot,
                        combo('accuracy'),
                        combo('coverage'),
                        combo('avg_set_size'),
                        combo('mrtl'),
                        combo('r1cr'),
                        combo('mrr'),
                        combo('rtc'),
                        combo('esr'),
                    ]
                    writer.writerow(row)

    print(f"🧾 CSV summary saved to: {csv_filename}")
    return csv_filename


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_final_experiment(embedding_file: str = None):
    """Run the complete final experiment pipeline"""
    print("🎯 FINAL CONFORMAL PREDICTION EXPERIMENT")
    print("="*70)
    print("This experiment will:")
    print("✅ Compare THREE methods: THR vs APS vs RAPS")
    print("✅ Test all K-shot values with robust trial counts")
    print("✅ Calculate comprehensive metrics including rank-based measures")
    print("✅ Generate final analysis and visualizations")
    print(f"⚙️  Classifier backbone: {CLASSIFIER_NAME}")
    print()
    
    # Step 2: Run full-scale experiment
    all_results = run_method_comparison_experiment(embedding_file=embedding_file)
    if all_results is None:
        print("❌ Experiment failed - no embeddings file found")
        return None
    
    # Step 3 & 4: Analysis and outputs
    summary_data = generate_summary_table(all_results)
    plot_filename = create_visualizations(all_results)
    results_filename = save_results(all_results, summary_data, embedding_file=embedding_file)
    csv_filename = save_summary_csv(summary_data, embedding_file=embedding_file)
    
    print("\n🎉 FINAL EXPERIMENT COMPLETED!")
    print("="*50)
    print("📊 Key Findings:")
    
    # Quick comparison summary
    simple_k10 = summary_data.get('simple', {}).get(10, {})
    raps_k10 = summary_data.get('raps', {}).get(10, {})
    
    if simple_k10 and raps_k10:
        print(f"📈 K=10 Comparison:")
        print(f"  Simple_CP: Size={simple_k10['avg_set_size']['mean']:.1f}, Coverage={simple_k10['coverage']['mean']:.3f}")
        print(f"  RAPS:      Size={raps_k10['avg_set_size']['mean']:.1f}, Coverage={raps_k10['coverage']['mean']:.3f}")
        print(f"  📊 RAPS is more conservative (larger sets, higher coverage)")
        print(f"  📊 Simple_CP is more efficient (smaller sets, target coverage)")
    
    print(f"\n📁 Files generated:")
    print(f"  📊 Plot: {plot_filename}")
    print(f"  💾 Data: {results_filename}")
    print(f"  🧾 CSV:  {csv_filename}")
    
    return all_results, summary_data


# =============================================================================
# COLAB CONVENIENCE FUNCTIONS
# =============================================================================

if IN_COLAB:
    def print_instructions():
        """Print Colab usage instructions"""
        print("📖 GOOGLE COLAB INSTRUCTIONS")
        print("="*40)
        print("1️⃣ Upload embeddings: upload_embeddings_colab()")
        print("2️⃣ Run experiment: run_final_experiment()")
        print("3️⃣ View results and plots!")

if __name__ == "__main__":
    if IN_COLAB:
        print("🚀 FINAL EXPERIMENT - GOOGLE COLAB VERSION")
        print_instructions()
    else:
        # Run directly if local
        run_final_experiment()