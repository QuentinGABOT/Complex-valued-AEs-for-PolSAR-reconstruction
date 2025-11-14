# Standard library imports
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third-party imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from scipy.signal import convolve2d
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.linalg import eigh
from skimage import exposure
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

# Local imports
from cvnn.data import get_dataset_type_from_config
from cvnn.utils import setup_logging

logger = setup_logging(__name__)


def pauli_transform(SAR_img: np.ndarray) -> np.ndarray:
    """
    Apply Pauli decomposition to PolSAR data.

    Args:
        SAR_img: Complex SAR image with shape (3, height, width) representing [HH, HV, VV]

    Returns:
        Pauli decomposed image with shape (3, height, width) representing [k1, k2, k3]

    Raises:
        ValueError: If input shape is invalid
    """
    if SAR_img.shape[0] == 4:
        SAR_img = np.stack(
            (
                SAR_img[0, :, :],  # HH
                (SAR_img[1, :, :] + SAR_img[1, :, :])/2,  # HV
                SAR_img[3, :, :],  # VV (assuming 4th channel is VV)
            ),
            axis=0,
        )
    elif SAR_img.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (HH, HV, VV), got {SAR_img.shape[0]}")

    S_HH = SAR_img[0, :, :]
    S_HV = SAR_img[1, :, :]
    S_VV = SAR_img[2, :, :]

    # Pauli basis vectors: k1 = (S_HH - S_VV)/√2, k2 = 2*S_HV/√2, k3 = (S_HH + S_VV)/√2
    sqrt_2_inv = 1.0 / np.sqrt(2.0)
    return sqrt_2_inv * np.stack(
        (
            S_HH - S_VV,
            2 * S_HV,
            S_HH + S_VV,
        ),
        dtype=np.complex64,
    )


def krogager_transform(SAR_img: np.ndarray) -> np.ndarray:
    """
    Apply Krogager decomposition to PolSAR data.

    Args:
        SAR_img: Complex SAR image with shape (3, height, width) representing [HH, HV, VV]

    Returns:
        Krogager decomposed image with shape (3, height, width) representing [kd, kh, ks]

    Raises:
        ValueError: If input shape is invalid
    """
    if SAR_img.shape[0] == 4:
        SAR_img = np.stack(
            (
                SAR_img[0, :, :],  # HH
                (SAR_img[1, :, :] + SAR_img[1, :, :])/2,  # HV
                SAR_img[3, :, :],  # VV (assuming 4th channel is VV)
            ),
            axis=0,
        )
    if SAR_img.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (HH, HV, VV), got {SAR_img.shape[0]}")

    S_HH = SAR_img[0, :, :]
    S_HV = SAR_img[1, :, :]
    S_VV = SAR_img[2, :, :]

    # Krogager decomposition: sphere, diplane, helix components
    S_RR = 1j * S_HV + 0.5 * (S_HH - S_VV)
    S_LL = 1j * S_HV - 0.5 * (S_HH - S_VV)
    S_RL = 0.5j * (S_HH + S_VV)  # Fixed coefficient

    return np.stack(
        (
            np.minimum(np.abs(S_RR), np.abs(S_LL)),  # kd: sphere component
            np.abs(np.abs(S_RR) - np.abs(S_LL)),  # kh: diplane component
            np.abs(S_RL),  # ks: helix component
        ),
        dtype=np.float32,  # Real-valued output
    )



def exp_amplitude_transform(tensor: Union[np.ndarray, torch.Tensor], dataset_name: str) -> torch.Tensor:
    """
    Apply exponential amplitude transformation to enhance SAR image visualization.

    This transformation maps normalized amplitudes [0,1] to a physically meaningful
    range [m, M] in dB scale for better contrast.

    Args:
        tensor: Input complex SAR tensor (numpy array or torch tensor)

    Returns:
        Transformed complex tensor with enhanced amplitude range

    Raises:
        ValueError: If input tensor is not complex-valued
    """
    # Convert to torch tensor if needed
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if not tensor.dtype.is_complex:
        raise ValueError(f"Expected complex tensor, got {tensor.dtype}")

    # Enhanced parameter range for better SAR visualization
    if dataset_name in ["ALOSDataset", "PolSFDataset"]:
        m = 0.009
        M = 0.992
    elif dataset_name == "Sethi":
        m = 4.7*10**-10
        M = 3.3*10**-8
    elif dataset_name == "Bretigny":
        m = 0.015
        M = 2.895

    amplitude = torch.abs(tensor)
    phase = torch.angle(tensor)

    # More robust log transformation
    log_M = safe_log(M, base=10)
    log_m = safe_log(m, base=10)

    # Enhanced amplitude transformation with better numerical stability
    inv_transformed_amplitude = torch.clamp(
        torch.exp(((log_M - log_m) * amplitude + log_m) * np.log(10)),
        min=0.0,
        max=1e9,  # Prevent overflow
    )

    # Recombine amplitude and phase
    new_tensor = inv_transformed_amplitude * torch.exp(1j * phase)
    return new_tensor


def equalize(
    image: np.ndarray, p2: Optional[float] = None, p98: Optional[float] = None
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Automatically adjust contrast of the SAR image using histogram equalization.

    Args:
        image: Input SAR image (intensity or amplitude)
        p2: Lower percentile for clipping (if None, computed automatically)
        p98: Upper percentile for clipping (if None, computed automatically)

    Returns:
        Tuple of (equalized_image, (p2_used, p98_used))

    Raises:
        ValueError: If input image is empty
    """
    if image.size == 0:
        raise ValueError("Input image cannot be empty")

    # Convert to log scale for better visualization
    img_log = safe_log(np.abs(image), base=10)

    # Compute percentiles if not provided
    if p2 is None or p98 is None:
        p2_computed, p98_computed = np.percentile(img_log, (2, 98))
        p2_final = p2 if p2 is not None else p2_computed
        p98_final = p98 if p98 is not None else p98_computed
    else:
        p2_final, p98_final = p2, p98

    # Ensure valid range
    if p2_final >= p98_final:
        logger.warning(
            f"Invalid percentile range: p2={p2_final}, p98={p98_final}. Using image min/max."
        )
        p2_final, p98_final = float(np.min(img_log)), float(np.max(img_log))

    # Rescale intensity (type checker issue with scikit-image)
    img_rescaled = exposure.rescale_intensity(
        img_log,
        in_range=(p2_final, p98_final),  # type: ignore
        out_range=(0, 1),  # type: ignore
    )

    # Convert to uint8
    img_final = np.round(img_rescaled * 255).astype(np.uint8)

    return img_final, (p2_final, p98_final)


def angular_distance(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Compute the angular distance between two complex-valued images.

    The angular distance measures the phase difference between corresponding pixels
    in two complex images, with results normalized to the range [-π, π].

    Args:
        image1: First complex-valued image (numpy array)
        image2: Second complex-valued image (numpy array)

    Returns:
        Angular distance array with values in [-π, π]

    Raises:
        ValueError: If input arrays have different shapes or are not complex-valued
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Input shapes must match: {image1.shape} vs {image2.shape}")

    if not (np.iscomplexobj(image1) and np.iscomplexobj(image2)):
        raise ValueError("Both input images must be complex-valued")

    # Compute phase difference and normalize to [-π, π]
    diff = np.angle(image1) - np.angle(image2) + np.pi
    angular_dist = np.mod(diff, 2 * np.pi) - np.pi
    return angular_dist


def plot_phase(image: np.ndarray) -> np.ndarray:
    """
    Convert phase information to displayable format normalized to [0, 255].

    Args:
        image: Complex-valued SAR image array

    Returns:
        Phase image normalized to uint8 range [0, 255]

    Raises:
        ValueError: If input is not complex-valued
    """
    if not np.iscomplexobj(image):
        raise ValueError("Input image must be complex-valued")

    phase_image = np.angle(image)  # Phase in [-π, π)
    # Normalize phase to [0, 1]
    normalized_phase = (phase_image + np.pi) / (2 * np.pi)
    # Scale to [0, 255] and convert to integer
    scaled_phase = np.round(normalized_phase * 255).astype(np.uint8)
    return scaled_phase


def plot_angular_distance(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Compute and visualize angular distance between two complex images.

    Args:
        image1: First complex-valued SAR image
        image2: Second complex-valued SAR image

    Returns:
        Angular distance image normalized to uint8 range [0, 255]

    Raises:
        ValueError: If images have different shapes or are not complex-valued
    """
    ang_distance_image = angular_distance(image1, image2)
    # Normalize angular distance from [-π, π] to [0, 1]
    normalized_ang_distance_image = (ang_distance_image + np.pi) / (2 * np.pi)
    # Scale to [0, 255] and convert to integer
    scaled_ang_distance_image = np.round(normalized_ang_distance_image * 255).astype(
        np.uint8
    )
    return scaled_ang_distance_image


def safe_log(
    x: Union[float, np.ndarray], base: float = 10, eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Compute logarithm safely by avoiding log(0) and negative values.

    Args:
        x: Input value(s) to compute logarithm of
        base: Logarithm base (10 for log10, math.e for natural log)
        eps: Minimum value to clamp inputs to avoid numerical issues

    Returns:
        Logarithm of the input with the specified base

    Raises:
        ValueError: If base is invalid
    """
    if base <= 0 or base == 1:
        raise ValueError(f"Invalid logarithm base: {base}")

    x_safe = np.clip(x, eps, None)
    if base == 10:
        return np.log10(x_safe)
    elif base == math.e:
        return np.log(x_safe)
    else:
        # Change of base formula: log_b(x) = ln(x) / ln(b)
        return np.log(x_safe) / np.log(base)

def _compute_h_alpha_coords(fullsamples: np.ndarray, son: int = 7, eps: float = 1e-10):
    """
    Vectorized Cloude–Pottier H (entropy ∈ [0,1]) and α (deg ∈ [0,90])
    on sliding son×son window.
    """
    H, W, p = fullsamples.shape
    H_out, W_out = H - (son - 1), W - (son - 1)

    kernel = np.ones((son, son), dtype=fullsamples.dtype)
    cov = np.empty((H_out, W_out, p, p), dtype=complex)
    for i in range(p):
        for j in range(p):
            prod = fullsamples[..., i] * np.conj(fullsamples[..., j])
            cov[..., i, j] = convolve2d(prod, kernel, mode="valid") / (son ** 2)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    p_vec = eigvals / eigvals.sum(axis=-1, keepdims=True)

    # Entropy normalized by log(3)
    H_mat = -np.sum(p_vec * (np.log(p_vec + eps) / np.log(3)), axis=-1)
    H_mat = np.clip(H_mat, 0, 1)

    alpha_i = np.arccos(np.clip(np.abs(eigvecs[..., 0, :]), 0, 1))
    alpha_mat = np.sum(p_vec * alpha_i, axis=-1) * (180.0 / np.pi)
    alpha_mat = np.clip(alpha_mat, 0, 90)

    return H_mat, alpha_mat

def _draw_halpha_zones(ax, class_info: Dict[int, Dict[str, str]]):
    """
    Draw the 9 standard Cloude–Pottier H–α zones as translucent rectangles.
    Axes: x = Entropy H ∈ [0,1], y = α ∈ [0,90] (degrees).
    Colors taken from class_info[id]["color"].
    """
    # Entropy band edges
    H_L, H_M, H_H, H_MAX = 0.0, 0.5, 0.9, 1.0

    # Alpha thresholds per band (deg)
    # Low entropy: 0–0.5
    th_L = [0.0, 42.5, 47.5, 90.0]     # → classes [9, 8, 7]
    # Medium entropy: 0.5–0.9
    th_M = [0.0, 40.0, 50.0, 90.0]     # → classes [6, 5, 4]
    # High entropy: 0.9–1.0
    th_H = [0.0, 40.0, 55.0, 90.0]           # → classes [3, 2, 1]

    zones = [
        # (Hmin, Hmax, αmin, αmax, class_id)
        # Low entropy band
        (H_L, H_M, th_L[0], th_L[1], 9),
        (H_L, H_M, th_L[1], th_L[2], 8),
        (H_L, H_M, th_L[2], th_L[3], 7),
        # Medium entropy band
        (H_M, H_M + (H_H - H_M), th_M[0], th_M[1], 6),
        (H_M, H_H,                 th_M[1], th_M[2], 5),
        (H_M, H_H,                 th_M[2], th_M[3], 4),
        # High entropy band
        (H_H, H_MAX, th_H[0], th_H[1], 3),
        (H_H, H_MAX, th_H[1], th_H[2], 2),
        (H_H, H_MAX, th_H[2], th_H[3], 1),
    ]

    # Draw rectangles
    for Hmin, Hmax, amin, amax, cid in zones:
        color = class_info.get(cid, {}).get("color", "lightgray")
        rect = Rectangle(
            (Hmin, amin), Hmax - Hmin, amax - amin,
            facecolor=color, edgecolor=color, alpha=0.12, linewidth=0.8, zorder=0
        )
        ax.add_patch(rect)
        # annotate class id
        ax.text(
            (Hmin + Hmax) / 2.0, (amin + amax) / 2.0, str(cid),
            ha="center", va="center", fontsize=8, alpha=0.6, zorder=1
        )    
        
    for x in [0.5, 0.9]:
        ax.axvline(x, ls="--", lw=0.8, alpha=0.4, zorder=2)
    for y in [40.0, 42.5, 47.5, 50.0, 55.0]:
        ax.axhline(y, ls="--", lw=0.8, alpha=0.4, zorder=2)


    # Gridlines at canonical boundaries

def _compute_classes_h_alpha(fullsamples: np.ndarray, son: int = 7) -> np.ndarray:
    """
    Vectorized H–α classification using convolution and batched eigendecomposition.
    """
    H, W, _ = fullsamples.shape
    H_out, W_out = H - (son - 1), W - (son - 1)
    H_mat, alpha_mat = _compute_h_alpha_coords(fullsamples, son=son)
    # 5) Vectorized classification rules
    classes = np.zeros((H_out, W_out), dtype=int)

    # H <= 0.5
    m = H_mat <= 0.5
    classes[m & (alpha_mat <= 42.5)] = 9
    classes[m & (alpha_mat > 42.5) & (alpha_mat <= 47.5)] = 8
    classes[m & (alpha_mat > 47.5)] = 7

    # 0.5 < H <= 0.9
    m = (H_mat > 0.5) & (H_mat <= 0.9)
    classes[m & (alpha_mat <= 40)] = 6
    classes[m & (alpha_mat > 40) & (alpha_mat <= 50)] = 5
    classes[m & (alpha_mat > 50)] = 4

    # 0.9 < H <= 1.0
    m = (H_mat > 0.9) & (H_mat <= 1.0)
    classes[m & (alpha_mat <= 55)] = 2
    classes[m & (alpha_mat > 55)] = 1

    return classes


def h_alpha(fullsamples: np.ndarray, son: int = 7) -> np.ndarray:
    """
    Wrapper for optimized H–α classification.
    """
    if fullsamples.ndim != 3 or fullsamples.shape[2] != 3:
        raise ValueError(f"Expected shape (H, W, 3), got {fullsamples.shape}")
    return _compute_classes_h_alpha(fullsamples, son)


def plot_losses(
    train_losses: Sequence[float],
    valid_losses: Optional[Sequence[float]] = None,
    savepath: Optional[Union[str, Path]] = None,
    title: str = "Loss Curve",
) -> None:
    """Plot train vs. validation loss over epochs."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    if valid_losses is not None:
        plt.plot(epochs, valid_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    logger.info(f"Loss plot saved to {savepath}")
    if wandb.run and savepath:
        wandb.log(
            {
                "plots/loss_curve": wandb.Image(str(savepath), caption=title),
            }
        )


def show_reconstructions(
    original_image: Union[np.ndarray, torch.Tensor],
    reconstructed_image: Union[np.ndarray, torch.Tensor],
    wandb_log: bool,
    logdir: Union[str, Path],
    cfg: Dict[str, Any],
) -> None:
    """
    Show original and reconstructed images as separate figures for better organization.
    Only computes PolSAR-specific visualizations for PolSAR datasets.
    """
    # Convert to numpy arrays if they are not already
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if isinstance(reconstructed_image, torch.Tensor):
        reconstructed_image = reconstructed_image.cpu().numpy()

    # Check dataset type to determine which visualizations to compute
    dataset_type = get_dataset_type_from_config(cfg)

    num_channels = original_image.shape[0]
    channels = ["HH", "HV", "VV"]

    img_dataset, img_gen = (
        exp_amplitude_transform(original_image, cfg["data"]["dataset"]["name"]).numpy(),
        exp_amplitude_transform(reconstructed_image, cfg["data"]["dataset"]["name"]).numpy(),
    )

    img_dataset_trans = img_dataset.transpose(1, 2, 0)
    img_gen_trans = img_gen.transpose(1, 2, 0)

    # Dictionary to store all figure paths for wandb logging
    figure_paths = {}

    # Only compute PolSAR-specific decompositions for PolSAR datasets
    if dataset_type == "polsar":
        polsar_metrics = {}

        pauli_img_dataset = pauli_transform(img_dataset).transpose(1, 2, 0)
        pauli_img_gen = pauli_transform(img_gen).transpose(1, 2, 0)

        krogager_img_dataset = krogager_transform(img_dataset).transpose(1, 2, 0)
        krogager_img_gen = krogager_transform(img_gen).transpose(1, 2, 0)

        # 1. Pauli decomposition comparison
        fig_pauli, axes_pauli = plt.subplots(1, 2, figsize=(12, 5))
        eq_dataset, (p2, p98) = equalize(pauli_img_dataset)
        axes_pauli[0].imshow(eq_dataset, origin="lower")
        axes_pauli[0].set_title("Original - Pauli Decomposition")
        axes_pauli[0].axis("off")

        eq_generated, _ = equalize(pauli_img_gen, p2=p2, p98=p98)
        axes_pauli[1].imshow(eq_generated, origin="lower")
        axes_pauli[1].set_title("Reconstructed - Pauli Decomposition")
        axes_pauli[1].axis("off")

        pauli_path = f"{logdir}/pauli_decomposition.png"
        fig_pauli.suptitle("Pauli Decomposition Comparison", fontsize=14)

        fig_pauli.savefig(pauli_path, bbox_inches="tight", dpi=150)
        plt.close(fig_pauli)
        figure_paths["pauli"] = pauli_path
        logger.info(f"Pauli decomposition comparison saved to {pauli_path}")

        # 2. Krogager decomposition comparison
        fig_krogager, axes_krogager = plt.subplots(1, 2, figsize=(12, 5))
        eq_dataset, (p2, p98) = equalize(krogager_img_dataset)
        axes_krogager[0].imshow(eq_dataset, origin="lower")
        axes_krogager[0].set_title("Original - Krogager Decomposition")
        axes_krogager[0].axis("off")

        eq_generated, _ = equalize(krogager_img_gen, p2=p2, p98=p98)
        axes_krogager[1].imshow(eq_generated, origin="lower")
        axes_krogager[1].set_title("Reconstructed - Krogager Decomposition")
        axes_krogager[1].axis("off")

        krogager_path = f"{logdir}/krogager_decomposition.png"
        fig_krogager.suptitle("Krogager Decomposition Comparison", fontsize=14)

        fig_krogager.savefig(krogager_path, bbox_inches="tight", dpi=150)
        plt.close(fig_krogager)
        figure_paths["krogager"] = krogager_path
        logger.info(f"Krogager decomposition comparison saved to {krogager_path}")

        # 3. Amplitude and angular distance analysis

        fig_distance, axes_distance = plt.subplots(1, 2, figsize=(12, 5))

        # Increase horizontal space between the two subplots to avoid overlapping y-labels
        fig_distance.subplots_adjust(wspace=0.35)

        # Amplitude difference histogram
        mse_values = (np.abs(img_dataset_trans) - np.abs(img_gen_trans)).flatten()
        q5, q95 = np.percentile(mse_values, [5, 95])
        filtered_data = mse_values[(mse_values > q5) & (mse_values < q95)]

        axes_distance[0].hist(filtered_data, bins=100, alpha=0.75)
        axes_distance[0].set_title("Amplitude Difference Histogram")
        axes_distance[0].set_xlabel("Amplitude Difference Value")
        axes_distance[0].set_ylabel("Frequency", labelpad=8)

        # Angular distance histogram
        axes_distance[1].hist(
            angular_distance(img_dataset_trans, img_gen_trans).flatten(),
            bins=100,
            alpha=0.75,
            range=(-1, 1),
        )
        axes_distance[1].set_title("Angular Distance Histogram")
        axes_distance[1].set_xlabel("Angular Distance (radians)")
        # move the second plot's y-axis to the right to avoid overlap and add padding
        axes_distance[1].set_ylabel("Frequency", labelpad=8)

        distance_path = f"{logdir}/distance_analysis.png"
        fig_distance.suptitle("Distance Analysis", fontsize=14)

        fig_distance.savefig(distance_path, bbox_inches="tight", dpi=150)
        plt.close(fig_distance)
        figure_paths["distance"] = distance_path
        logger.info(f"Distance analysis saved to {distance_path}")

        # 4. H-alpha classification comparison
        h_alpha_class_info = {
            1: {"color": "green", "name": "Complex structures"},
            2: {"color": "yellow", "name": "Random anisotropic scatterers"},
            4: {"color": "blue", "name": "Double reflection propagation effects"},
            5: {"color": "pink", "name": "Anisotropic particles"},
            6: {"color": "purple", "name": "Random surfaces"},
            7: {"color": "red", "name": "Dihedral reflector"},
            8: {"color": "brown", "name": "Dipole"},
            9: {"color": "gray", "name": "Bragg surface"},
        }
        h_alpha_class_colors = {k: v["color"] for k, v in h_alpha_class_info.items()}
        h_alpha_cmap = ListedColormap([i for i in h_alpha_class_colors.values()])
        h_alpha_bounds = list(h_alpha_class_colors.keys())
        h_alpha_norm = BoundaryNorm(h_alpha_bounds, len(h_alpha_class_colors))
        h_alpha_patches = [
            mpatches.Patch(color=h_alpha_class_info[i]["color"],
                          label=f"{i}: {h_alpha_class_info[i]['name']}")
            for i in h_alpha_class_info
        ]

        h_alpha_original = h_alpha(pauli_img_dataset)
        h_alpha_gen = h_alpha(pauli_img_gen)

        fig_halpha, axes_halpha = plt.subplots(1, 3, figsize=(18, 5))

        # Original H-alpha
        im1 = axes_halpha[0].imshow(
            h_alpha_original, origin="lower", cmap=h_alpha_cmap, norm=h_alpha_norm
        )
        axes_halpha[0].set_title("Original H-alpha Classification")
        axes_halpha[0].axis("off")

        # Reconstructed H-alpha
        im2 = axes_halpha[1].imshow(h_alpha_gen, origin="lower", cmap=h_alpha_cmap, norm=h_alpha_norm)
        axes_halpha[1].set_title("Reconstructed H-alpha Classification")
        axes_halpha[1].axis("off")

        # Confusion matrix
        _plot_confusion_matrix(
            axes_halpha[2], h_alpha_original, h_alpha_gen, h_alpha_class_colors
        )

        # Add legend
        fig_halpha.legend(handles=h_alpha_patches, bbox_to_anchor=(1.15, 0.8), loc="upper left")

        halpha_path = f"{logdir}/h_alpha_classification.png"
        fig_halpha.suptitle("H-alpha Classification Comparison", fontsize=14)

        fig_halpha.savefig(halpha_path, bbox_inches="tight", dpi=150)
        plt.close(fig_halpha)
        figure_paths["h_alpha"] = halpha_path
        logger.info(f"H-alpha classification comparison saved to {halpha_path}")

        # Compute comprehensive H-alpha classification metrics
        h_alpha_metrics = compute_h_alpha_metrics(
            h_alpha_original, h_alpha_gen, list(h_alpha_class_colors.keys())
        )
        polsar_metrics["h_alpha_metrics"] = h_alpha_metrics

        logger.info(
            f"Accuracy between the H_alpha labels is: {100 * h_alpha_metrics['accuracy']:.3f}%"
        )
        logger.info(f"H-alpha Cohen's Kappa: {h_alpha_metrics['cohen_kappa']:.3f}")
        logger.info(f"H-alpha F1-score (macro): {h_alpha_metrics['f1_macro']:.3f}")
        logger.info(
            f"H-alpha F1-score (weighted): {h_alpha_metrics['f1_weighted']:.3f}"
        )

        # === H–α misclassification shift plot (only misclassified pixels) ===
        son = 7
        H_orig, A_orig = _compute_h_alpha_coords(pauli_img_dataset, son=son)
        H_gen,  A_gen  = _compute_h_alpha_coords(pauli_img_gen, son=son)

        assert h_alpha_original.shape == h_alpha_gen.shape == H_orig.shape == H_gen.shape
        miscls_mask = (h_alpha_original != h_alpha_gen)
        idx_y, idx_x = np.where(miscls_mask)

        max_segments = 5000
        if idx_x.size > max_segments:
            sel = np.random.RandomState(0).choice(idx_x.size, size=max_segments, replace=False)
            idx_x, idx_y = idx_x[sel], idx_y[sel]

        # Now H on x-axis, α on y-axis
        segs = np.stack([
            np.stack([H_orig[idx_y, idx_x], A_orig[idx_y, idx_x]], axis=-1),
            np.stack([H_gen[idx_y, idx_x],  A_gen[idx_y, idx_x]],  axis=-1),
        ], axis=1)

        fig_shift, ax_shift = plt.subplots(1, 1, figsize=(7, 6))
        ax_shift.set_title("H–α Misclassification Shifts (orig → recon)")
        ax_shift.set_xlabel("Entropy H")
        ax_shift.set_ylabel("Scattering angle α (degrees)")
        ax_shift.set_xlim(0, 1)
        ax_shift.set_ylim(0, 90)
        ax_shift.grid(True, ls="--", alpha=0.4)

        # draw zones behind everything
        _draw_halpha_zones(ax_shift, h_alpha_class_info)

        def _compute_physical_boundary():
            """
            Compute the physically feasible boundary of the Cloude–Pottier H–α plane.
            Returns (entropy_all, alpha_all_deg).
            """
            m2 = np.arange(0, 1.01, 0.01)
            entropy2 = []
            alpha2 = []
            entropy3 = []
            alpha3 = []

            for i in range(101):
                T = np.array([[1, 0, 0],
                            [0, m2[i], 0],
                            [0, 0, m2[i]]])
            
                D, V = np.linalg.eig(T)
                tr = np.sum(D)
                P = D / tr + np.finfo(float).eps  # Avoid log(0)
            
                alpha2_val = np.sum(P * np.arccos(np.abs(V[0, :])))
                entropy2_val = -np.sum(P * np.log10(P)) / np.log10(3)
            
                alpha2.append(alpha2_val)
                entropy2.append(entropy2_val)
            
                if i < 50:
                    T2 = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 2*m2[i]]])
                else:
                    T2 = np.array([[2*m2[i]-1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
            
                D2, V2 = np.linalg.eig(T2)
                tr2 = np.sum(D2)
                P2 = D2 / tr2 + np.finfo(float).eps
            
                alpha3_val = np.sum(P2 * np.arccos(np.abs(V2[0, :])))
                entropy3_val = -np.sum(P2 * np.log10(P2)) / np.log10(3)
            
                alpha3.append(alpha3_val)
                entropy3.append(entropy3_val)
            return alpha2, entropy2, alpha3, entropy3

        alpha2, entropy2, alpha3, entropy3 = _compute_physical_boundary()
        ax_shift.plot(entropy2, np.degrees(alpha2),
                    'k-', linewidth=2.0,
                    label='Physical boundary', zorder=4)
        ax_shift.plot(entropy3, np.degrees(alpha3),
                    'k-', linewidth=2.0,
                    zorder=4)

        ax_shift.legend(frameon=True, loc="upper left")

        if segs.shape[0] == 0:
            ax_shift.text(0.5, 0.5, "No misclassified pixels", ha="center", va="center", transform=ax_shift.transAxes)
        else:
            lc = LineCollection(segs, linewidths=0.5, alpha=0.35)
            ax_shift.add_collection(lc)

            ax_shift.scatter(H_orig[idx_y, idx_x], A_orig[idx_y, idx_x],
                            s=6, marker='o', c='red',  label='Original', alpha=0.25)
            ax_shift.scatter(H_gen[idx_y, idx_x],  A_gen[idx_y, idx_x],
                            s=6, marker='o', c='blue', label='Reconstructed', alpha=0.25)
            ax_shift.legend(frameon=True, loc="upper left")

        halpha_shift_path = f"{logdir}/h_alpha_misclassification_shifts.png"
        fig_shift.tight_layout()
        fig_shift.savefig(halpha_shift_path, bbox_inches="tight", dpi=150)
        plt.close(fig_shift)
        figure_paths["h_alpha_shifts"] = halpha_shift_path
        logger.info(f"H–α misclassification shift plot saved to {halpha_shift_path}")
        
        # 5. Cameron Classification Visualization
        logger.info("Computing Cameron classification comparison...")
        
        # Define Cameron class colors and names (classes 1-11)
        cameron_class_info = {
            1: {"color": "red", "name": "Non-reciprocal"},
            2: {"color": "orange", "name": "Asymmetric"},  
            3: {"color": "yellow", "name": "Left helix"},
            4: {"color": "green", "name": "Right helix"},
            5: {"color": "blue", "name": "Symmetric"},
            6: {"color": "purple", "name": "Trihedral"},
            7: {"color": "brown", "name": "Dihedral"},
            8: {"color": "pink", "name": "Dipole"},
            9: {"color": "gray", "name": "Cylinder"},
            10: {"color": "olive", "name": "Narrow dihedral"},
            11: {"color": "cyan", "name": "Quarter-wave"},
        }
        cameron_class_colors = {k: v["color"] for k, v in cameron_class_info.items()}
        cameron_cmap = ListedColormap([i for i in cameron_class_colors.values()])
        cameron_bounds = list(cameron_class_colors.keys())
        cameron_norm = BoundaryNorm(cameron_bounds, len(cameron_class_colors))
        cameron_patches = [
            mpatches.Patch(color=cameron_class_info[i]["color"], 
                          label=f"{i}: {cameron_class_info[i]['name']}")
            for i in cameron_class_info
        ]

        # Compute Cameron classifications
        cameron_original = cameron(img_dataset)
        cameron_gen = cameron(img_gen)

        fig_cameron, axes_cameron = plt.subplots(1, 3, figsize=(18, 5))

        # Original Cameron
        im1_cameron = axes_cameron[0].imshow(
            cameron_original, origin="lower", cmap=cameron_cmap, norm=cameron_norm
        )
        axes_cameron[0].set_title("Original Cameron Classification")
        axes_cameron[0].axis("off")

        # Reconstructed Cameron
        im2_cameron = axes_cameron[1].imshow(
            cameron_gen, origin="lower", cmap=cameron_cmap, norm=cameron_norm
        )
        axes_cameron[1].set_title("Reconstructed Cameron Classification")
        axes_cameron[1].axis("off")

        # Confusion matrix for Cameron
        _plot_confusion_matrix(
            axes_cameron[2], cameron_original, cameron_gen, cameron_class_colors
        )

        # Add legend
        fig_cameron.legend(handles=cameron_patches, bbox_to_anchor=(1.15, 0.8), loc="upper left")

        cameron_path = f"{logdir}/cameron_classification.png"
        fig_cameron.suptitle("Cameron Classification Comparison", fontsize=14)

        fig_cameron.savefig(cameron_path, bbox_inches="tight", dpi=150)
        plt.close(fig_cameron)
        figure_paths["cameron"] = cameron_path
        logger.info(f"Cameron classification comparison saved to {cameron_path}")

        # Compute comprehensive Cameron classification metrics
        cameron_metrics = compute_cameron_metrics(
            cameron_original, cameron_gen, list(cameron_class_colors.keys())
        )
        polsar_metrics["cameron_metrics"] = cameron_metrics

        logger.info(
            f"Accuracy between the Cameron labels is: {100 * cameron_metrics['accuracy']:.3f}%"
        )
        logger.info(f"Cameron Cohen's Kappa: {cameron_metrics['cohen_kappa']:.3f}")
        logger.info(f"Cameron F1-score (macro): {cameron_metrics['f1_macro']:.3f}")
        logger.info(
            f"Cameron F1-score (weighted): {cameron_metrics['f1_weighted']:.3f}"
        )
    else:
        logger.info(
            f"Skipping PolSAR-specific visualizations for dataset type: {dataset_type}"
        )

    # Basic visualizations for all dataset types

    # Log to Weights & Biases if enabled (only for PolSAR datasets with generated figures)
    if wandb_log and dataset_type == "polsar":
        # Create wandb log dictionary with separate figures
        wandb_log_dict = {
            "plots/pauli_decomposition": wandb.Image(
                figure_paths["pauli"], caption="Pauli Decomposition Comparison"
            ),
            "plots/krogager_decomposition": wandb.Image(
                figure_paths["krogager"], caption="Krogager Decomposition Comparison"
            ),
            "plots/distance_analysis": wandb.Image(
                figure_paths["distance"],
                caption="Amplitude and Angular Distance Analysis",
            ),
            "plots/h_alpha_classification": wandb.Image(
                figure_paths["h_alpha"], caption="H-alpha Classification Comparison"
            ),
            "plots/cameron_classification": wandb.Image(
                figure_paths["cameron"], caption="Cameron Classification Comparison"
            ),
            "plots/h_alpha_misclassification_shifts": wandb.Image(
                figure_paths["h_alpha_shifts"],
                caption="H-alpha Misclassification Shifts",
            ),
        }

        # Add H-alpha classification metrics to wandb
        wandb_log_dict.update(
            {
                f"h_alpha/{key}": value
                for key, value in h_alpha_metrics.items()
                if key
                not in [
                    "per_class_metrics",
                    "confusion_matrix_raw",
                    "confusion_matrix_normalized",
                    "class_labels",
                ]
            }
        )

        # Add per-class metrics with proper wandb formatting
        for class_id, class_metrics in h_alpha_metrics["per_class_metrics"].items():
            for metric_name, metric_value in class_metrics.items():
                wandb_log_dict[f"h_alpha/class_{class_id}/{metric_name}"] = metric_value

        # Add Cameron classification metrics to wandb
        wandb_log_dict.update(
            {
                f"cameron/{key}": value
                for key, value in cameron_metrics.items()
                if key
                not in [
                    "per_class_metrics",
                    "confusion_matrix_raw",
                    "confusion_matrix_normalized",
                    "class_labels",
                ]
            }
        )

        # Add Cameron per-class metrics with proper wandb formatting
        for class_id, class_metrics in cameron_metrics["per_class_metrics"].items():
            for metric_name, metric_value in class_metrics.items():
                wandb_log_dict[f"cameron/class_{class_id}/{metric_name}"] = metric_value

        wandb.log(wandb_log_dict)
        # return h-alpha and cameron metrics
        return polsar_metrics
        #return {"h_alpha_metrics": {"accuracy":1}, "cameron_metrics": {"accuracy":1}}

    else:
        logger.info("WandB logging is disabled or not a PolSAR dataset. Skipping logging.")
        return {}


def _plot_confusion_matrix(
    ax, orig: np.ndarray, gen: np.ndarray, class_colors: Dict[int, str]
) -> None:
    """Plot normalized confusion matrix of H-alpha classes on the provided Axes."""
    cm = confusion_matrix(orig.flatten(), gen.flatten(), normalize="true").round(3)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2g",
        cmap="hot",
        ax=ax,
        xticklabels=[str(i) for i in class_colors.keys()],
        yticklabels=[str(i) for i in class_colors.keys()],
    )
    ax.set_xlabel("Reconstructed classes")
    ax.set_ylabel("Original classes")
    ax.set_title("Confusion Matrix")

def plot_segmentation_results(
    metrics: Dict[str, Any],
    savepath: str,
    title: str = "Segmentation Results",
    skip_sample_display: bool = False,
) -> None:
    """Plot enhanced segmentation evaluation results with prediction overlays and error analysis."""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # Determine layout based on whether we show samples
    if skip_sample_display:
        # Only metrics charts - use 2x3 layout
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    else:
        # Full layout with samples - use 4x5 layout
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

    # --- Row 1: Per-class performance metrics ---

    # IoU and Dice per class
    if skip_sample_display:
        ax1 = fig.add_subplot(gs[0, 0])
    else:
        ax1 = fig.add_subplot(gs[0, 0:2])

    if "iou_per_class" in metrics and "dice_per_class" in metrics:
        iou_per_class = metrics["iou_per_class"]
        dice_per_class = metrics["dice_per_class"]
        class_indices = list(range(len(iou_per_class)))

        x = np.arange(len(class_indices))
        width = 0.35

        ax1.bar(x - width / 2, iou_per_class, width, label="IoU", alpha=0.8)
        ax1.bar(x + width / 2, dice_per_class, width, label="Dice", alpha=0.8)
        ax1.set_title("IoU and Dice per Class")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"C{i}" for i in class_indices])

    # Precision, Recall, F1 per class
    if skip_sample_display:
        ax2 = fig.add_subplot(gs[0, 1])
    else:
        ax2 = fig.add_subplot(gs[0, 2:4])

    if all(
        key in metrics
        for key in ["precision_per_class", "recall_per_class", "f1_per_class"]
    ):
        precision = metrics["precision_per_class"]
        recall = metrics["recall_per_class"]
        f1 = metrics["f1_per_class"]

        x = np.arange(len(precision))
        width = 0.25

        ax2.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax2.bar(x, recall, width, label="Recall", alpha=0.8)
        ax2.bar(x + width, f1, width, label="F1", alpha=0.8)
        ax2.set_title("Precision, Recall, F1 per Class")
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"C{i}" for i in range(len(precision))])

    # Overall metrics summary
    if skip_sample_display:
        ax3 = fig.add_subplot(gs[0, 2])
    else:
        ax3 = fig.add_subplot(gs[0, 4])

    text_content = []
    metrics_to_show = [
        ("Pixel Accuracy", "pixel_accuracy"),
        ("Weighted Accuracy", "weighted_accuracy"),
        ("Mean IoU", "mean_iou"),
        ("Freq. Weighted IoU", "frequency_weighted_iou"),
        ("Mean Dice", "mean_dice"),
        ("F1 Macro", "f1_macro"),
        ("F1 Weighted", "f1_weighted"),
    ]

    for label, key in metrics_to_show:
        if key in metrics:
            text_content.append(f"{label}: {metrics[key]:.3f}")

    ax3.text(
        0.1,
        0.5,
        "\n".join(text_content),
        fontsize=10,
        transform=ax3.transAxes,
        verticalalignment="center",
    )
    ax3.set_title("Overall Metrics")
    ax3.axis("off")

    # --- Row 2: Confusion Matrix and Class Distribution ---

    # Confusion Matrix
    if skip_sample_display:
        ax4 = fig.add_subplot(gs[1, 0:2])
    else:
        ax4 = fig.add_subplot(gs[1, 0:2])

    if "confusion_matrix" in metrics:
        conf_matrix = np.array(metrics["confusion_matrix"])
        im = ax4.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        ax4.set_title("Confusion Matrix")

        # Add colorbar
        plt.colorbar(im, ax=ax4)

        # Add text annotations
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax4.text(
                    j,
                    i,
                    format(conf_matrix[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                )

        ax4.set_ylabel("True Label")
        ax4.set_xlabel("Predicted Label")

    # Class frequency distribution
    if skip_sample_display:
        ax5 = fig.add_subplot(gs[1, 2])
    else:
        ax5 = fig.add_subplot(gs[1, 2:4])

    if "class_frequencies" in metrics:
        class_freq = metrics["class_frequencies"]
        class_indices = list(range(len(class_freq)))
        ax5.bar(class_indices, class_freq, alpha=0.7)
        ax5.set_title("Class Distribution")
        ax5.set_xlabel("Class")
        ax5.set_ylabel("Frequency")
        ax5.set_xticks(class_indices)
        ax5.set_xticklabels([f"C{i}" for i in class_indices])

    # --- Conditional: Sample Predictions and Error Analysis (only if not skipping) ---
    if not skip_sample_display and all(
        key in metrics
        for key in ["sample_images", "sample_ground_truth", "sample_predictions"]
    ):
        sample_images = metrics["sample_images"]
        sample_gt = metrics["sample_ground_truth"]
        sample_pred = metrics["sample_predictions"]

        n_samples = min(len(sample_images), 5)

        for i in range(n_samples):
            # Row 3: Ground Truth vs Predictions (side by side)
            ax_gt = fig.add_subplot(gs[2, i])
            ax_pred = fig.add_subplot(gs[3, i])

            # Show ground truth
            if len(sample_gt[i].shape) == 2:  # 2D segmentation mask
                im_gt = ax_gt.imshow(sample_gt[i], cmap="viridis")
                ax_gt.set_title(f"Sample {i+1}: Ground Truth")
                ax_gt.axis("off")

                # Show prediction
                im_pred = ax_pred.imshow(sample_pred[i], cmap="viridis")
                ax_pred.set_title(f"Sample {i+1}: Prediction")
                ax_pred.axis("off")

                # Create error overlay (highlight misclassified pixels in red)
                error_mask = (sample_gt[i] != sample_pred[i]).astype(float)

                # Create an RGB overlay
                overlay = np.zeros((*sample_gt[i].shape, 3))
                # Background: normalized prediction
                pred_norm = sample_pred[i] / (sample_pred[i].max() + 1e-8)
                overlay[:, :, 0] = pred_norm  # Red channel for base prediction
                overlay[:, :, 1] = pred_norm  # Green channel for base prediction
                overlay[:, :, 2] = pred_norm  # Blue channel for base prediction

                # Highlight errors in bright red
                overlay[error_mask > 0] = [1.0, 0.0, 0.0]  # Bright red for errors

                ax_pred.imshow(overlay, alpha=0.7)

    plt.suptitle(title, fontsize=16, fontweight="bold")

    # Save the comprehensive visualization
    save_path = Path(savepath)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Enhanced segmentation results saved to {savepath}")

    # Only create separate focused visualizations if samples are displayed
    if not skip_sample_display:
        _plot_prediction_overlays(
            metrics, save_path.parent / f"{save_path.stem}_overlays.png"
        )
        _plot_error_analysis(metrics, save_path.parent / f"{save_path.stem}_errors.png")


def _plot_prediction_overlays(metrics: Dict[str, Any], savepath: Path) -> None:
    """Create focused side-by-side prediction overlay visualization."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not all(
        key in metrics
        for key in ["sample_images", "sample_ground_truth", "sample_predictions"]
    ):
        return

    sample_images = metrics["sample_images"]
    sample_gt = metrics["sample_ground_truth"]
    sample_pred = metrics["sample_predictions"]

    n_samples = min(len(sample_images), 5)

    fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 9))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        # Original image (if it's a complex image, show magnitude)
        img = sample_images[i]
        if np.iscomplexobj(img):
            img_display = np.abs(img)
            if img_display.ndim == 3:  # Multi-channel
                img_display = img_display[0]  # Take first channel
        else:
            if img.ndim == 3:  # Multi-channel
                img_display = img[0]  # Take first channel
            else:
                img_display = img

        axes[0, i].imshow(img_display, cmap="gray")
        axes[0, i].set_title(f"Sample {i+1}: Input")
        axes[0, i].axis("off")

        # Ground truth
        axes[1, i].imshow(sample_gt[i], cmap="viridis")
        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis("off")

        # Prediction
        axes[2, i].imshow(sample_pred[i], cmap="viridis")
        axes[2, i].set_title("Prediction")
        axes[2, i].axis("off")

    plt.suptitle("Prediction Overlays", fontsize=14, fontweight="bold")

    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Prediction overlays saved to {savepath}")


def _plot_error_analysis(metrics: Dict[str, Any], savepath: Path) -> None:
    """Create focused error analysis visualization."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not all(key in metrics for key in ["sample_ground_truth", "sample_predictions"]):
        return

    sample_gt = metrics["sample_ground_truth"]
    sample_pred = metrics["sample_predictions"]

    n_samples = min(len(sample_gt), 5)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        # Error mask (misclassified pixels)
        error_mask = (sample_gt[i] != sample_pred[i]).astype(float)

        # Show prediction with error overlay
        axes[0, i].imshow(sample_pred[i], cmap="viridis", alpha=0.7)
        axes[0, i].imshow(error_mask, cmap="Reds", alpha=0.5)
        axes[0, i].set_title(f"Sample {i+1}: Errors (Red)")
        axes[0, i].axis("off")

        # Show error mask only
        axes[1, i].imshow(error_mask, cmap="Reds")
        axes[1, i].set_title("Error Mask")
        axes[1, i].axis("off")

        # Calculate error percentage for this sample
        error_pct = (error_mask.sum() / error_mask.size) * 100
        axes[1, i].text(
            0.5,
            -0.1,
            f"Error: {error_pct:.1f}%",
            transform=axes[1, i].transAxes,
            ha="center",
        )

    plt.suptitle("Error Analysis", fontsize=14, fontweight="bold")

    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Error analysis saved to {savepath}")


def show_generated_samples(
    metrics: Dict[str, Any],
    logdir: Union[str, Path],
    wandb_log: bool = False,
) -> None:
    """Show generated samples and metrics."""
    import matplotlib.pyplot as plt

    # Simple metrics display
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    text_content = []
    if "generation_mse" in metrics:
        text_content.append(f"Generation MSE: {metrics['generation_mse']:.6f}")
    if "num_generated_samples" in metrics:
        text_content.append(f"Number of samples: {metrics['num_generated_samples']}")
    if "sample_shape" in metrics and metrics["sample_shape"] is not None:
        text_content.append(f"Sample shape: {metrics['sample_shape']}")

    ax.text(
        0.1,
        0.5,
        "\n".join(text_content),
        fontsize=14,
        transform=ax.transAxes,
        verticalalignment="center",
    )
    ax.set_title("Generation Metrics")
    ax.axis("off")

    savepath = Path(logdir) / "generation_metrics.png"

    plt.savefig(savepath)
    plt.close()
    logger.info(f"Generation metrics saved to {savepath}")

    if wandb_log and wandb.run:
        wandb.log(
            {
                "plots/generation_metrics": [
                    wandb.Image(str(savepath), caption="Generation Metrics")
                ]
            }
        )


def compute_h_alpha_metrics(
    h_alpha_original: np.ndarray,
    h_alpha_generated: np.ndarray,
    class_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for H-alpha classification comparison.

    Args:
        h_alpha_original: Original H-alpha classification map
        h_alpha_generated: Generated/reconstructed H-alpha classification map
        class_labels: List of class labels (defaults to H-alpha classes [1,2,4,5,6,7,8,9])

    Returns:
        Dictionary containing various classification metrics
    """
    if class_labels is None:
        class_labels = [
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
        ]  # Standard H-alpha classes (3 not possible)

    # Flatten arrays for sklearn metrics
    y_true = h_alpha_original.flatten()
    y_pred = h_alpha_generated.flatten()

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix (normalized and raw)
    cm_raw = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_normalized = confusion_matrix(
        y_true, y_pred, labels=class_labels, normalize="true"
    )

    # Per-class metrics
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )

    # Macro and micro averages
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, fscore_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=class_labels, average="weighted", zero_division=0
        )
    )

    # Additional metrics
    kappa = cohen_kappa_score(y_true, y_pred)

    # Try to compute Matthews correlation coefficient
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        # MCC not defined for some edge cases
        mcc = np.nan

    # Class distribution in original and generated
    unique_orig, counts_orig = np.unique(y_true, return_counts=True)
    unique_gen, counts_gen = np.unique(y_pred, return_counts=True)

    class_distribution_orig = {
        int(cls): int(count) for cls, count in zip(unique_orig, counts_orig)
    }
    class_distribution_gen = {
        int(cls): int(count) for cls, count in zip(unique_gen, counts_gen)
    }

    # Create comprehensive metrics dictionary
    metrics = {
        # Overall metrics
        "accuracy": float(accuracy),
        "cohen_kappa": float(kappa),
        "matthews_corrcoef": float(mcc) if not np.isnan(mcc) else None,
        # Macro averages
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(fscore_macro),
        # Micro averages
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(fscore_micro),
        # Weighted averages
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(fscore_weighted),
        # Per-class metrics
        "per_class_metrics": {
            int(class_labels[i]): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(fscore[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_labels))
        },
        # Class distributions
        "class_distribution_original": class_distribution_orig,
        "class_distribution_generated": class_distribution_gen,
        # Confusion matrices
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "class_labels": class_labels,
    }

    return metrics


def cameron_transform(SAR_img: np.ndarray) -> List[np.ndarray]:
    """
    Compute Cameron decomposition parameters from PolSAR image.
    
    This function implements the Cameron coherent polarimetric decomposition
    based on the method described in Cameron, Youssef, and Leung (1996).
    The decomposition extracts symmetric and asymmetric scattering components.
    
    Args:
        SAR_img: Complex SAR image with shape (3, H, W) for [HH, HV, VV] 
                 or (4, H, W) for [HH, HV, VH, VV]
        
    Returns:
        List of Cameron decomposition parameters:
        - For 3-pol data: 9 parameters [S_max1, S_max2, S_max4, S_min1, S_min2, S_min4, a, Tau, Psi_D]
        - For 4-pol data: 13 parameters [S_max1, S_max2, S_max3, S_max4, S_min1, S_min2, S_min3, S_min4, S_nr, a, Tau, Theta_rec, Psi_D]
        - For test compatibility: Always returns 13 parameters (padding with zeros for 3-pol)
        
    Raises:
        ValueError: If input doesn't have expected 3 or 4 channel format
    """
    if SAR_img.shape[0] not in [3, 4]:
        raise ValueError(f"Expected 3 channels, got {SAR_img.shape[0]}")
    
    is_full_pol = (SAR_img.shape[0] == 4)
    
    if is_full_pol:
        # 4-pol case: [HH, HV, VH, VV] 
        S_hh = SAR_img[0]
        S_hv = SAR_img[1]  
        S_vh = SAR_img[2]
        S_vv = SAR_img[3]
        
        # Compute norm of scattering vector (full-pol)
        a = np.sqrt(np.abs(S_hh)**2 + np.abs(S_hv)**2 + np.abs(S_vh)**2 + np.abs(S_vv)**2)
        
        # Determine Pauli parameters (full-pol)
        alpha = (S_hh + S_vv) / np.sqrt(2)
        beta = (S_hh - S_vv) / np.sqrt(2)
        gamma = (S_hv + S_vh) / np.sqrt(2)
        delta = (S_vh - S_hv) / np.sqrt(2)
        
        # Determine parameter x
        numerator = beta * np.conj(gamma) + np.conj(beta) * gamma
        denominator_sq = numerator**2 + (np.abs(beta)**2 - np.abs(gamma)**2)**2
        
        # Avoid division by zero
        valid_mask = denominator_sq > 1e-10
        sin_x = np.zeros_like(numerator, dtype=complex)
        cos_x = np.zeros_like(numerator, dtype=complex)
        
        sin_x[valid_mask] = numerator[valid_mask] / np.sqrt(denominator_sq[valid_mask])
        cos_x[valid_mask] = (np.abs(beta[valid_mask])**2 - np.abs(gamma[valid_mask])**2) / np.sqrt(denominator_sq[valid_mask])
        
        # Compute angle x
        x = np.zeros_like(sin_x, dtype=float)
        sin_x_real = np.real(sin_x)
        cos_x_real = np.real(cos_x)
        
        # Handle different quadrants
        mask1 = (sin_x_real >= 0)
        mask2 = (sin_x_real < 0) & (cos_x_real >= 0)
        mask3 = (sin_x_real < 0) & (cos_x_real < 0)
        
        x[mask1] = np.arccos(np.clip(cos_x_real[mask1], -1, 1))
        x[mask2] = np.arcsin(np.clip(sin_x_real[mask2], -1, 1))
        x[mask3] = -np.arcsin(np.clip(sin_x_real[mask3], -1, 1)) - np.pi
        
        # Determine DS (dominant symmetric component) - full-pol
        scalar = (S_hh * np.cos(x/2) + S_hv * np.sin(x/2) + S_vh * np.sin(x/2) - S_vv * np.cos(x/2)) / np.sqrt(2)
        
        DS_1 = (alpha + np.cos(x/2) * scalar) / np.sqrt(2)
        DS_2 = np.sin(x/2) * scalar / np.sqrt(2)
        DS_3 = np.sin(x/2) * scalar / np.sqrt(2)
        DS_4 = (alpha - np.cos(x/2) * scalar) / np.sqrt(2)
        
        # Normalize S_max
        S_max_norm = np.sqrt(np.abs(DS_1)**2 + np.abs(DS_2)**2 + np.abs(DS_3)**2 + np.abs(DS_4)**2)
        S_max_norm = np.maximum(S_max_norm, 1e-10)
        
        S_max1 = DS_1 / S_max_norm
        S_max2 = DS_2 / S_max_norm
        S_max3 = DS_3 / S_max_norm
        S_max4 = DS_4 / S_max_norm
        
        # Determine S_rec (reciprocal component) - full-pol
        S_rec1 = S_hh
        S_rec2 = (S_hv + S_vh) / 2
        S_rec3 = (S_hv + S_vh) / 2
        S_rec4 = S_vv
        
        # Determine DS_rec
        scalar_rec = (S_rec1 * np.cos(x/2) + S_rec2 * np.sin(x/2) + 
                      S_rec3 * np.sin(x/2) - S_rec4 * np.cos(x/2)) / np.sqrt(2)
        
        DS_rec1 = (alpha + np.cos(x/2) * scalar_rec) / np.sqrt(2)
        DS_rec2 = np.sin(x/2) * scalar_rec / np.sqrt(2)
        DS_rec3 = np.sin(x/2) * scalar_rec / np.sqrt(2)
        DS_rec4 = (alpha - np.cos(x/2) * scalar_rec) / np.sqrt(2)
        
        # Determine S_min
        S_min1_unnorm = S_rec1 - DS_rec1
        S_min2_unnorm = S_rec2 - DS_rec2
        S_min3_unnorm = S_rec3 - DS_rec3
        S_min4_unnorm = S_rec4 - DS_rec4
        
        S_min_norm = np.sqrt(np.abs(S_min1_unnorm)**2 + np.abs(S_min2_unnorm)**2 + 
                            np.abs(S_min3_unnorm)**2 + np.abs(S_min4_unnorm)**2)
        S_min_norm = np.maximum(S_min_norm, 1e-10)
        
        S_min1 = S_min1_unnorm / S_min_norm
        S_min2 = S_min2_unnorm / S_min_norm
        S_min3 = S_min3_unnorm / S_min_norm
        S_min4 = S_min4_unnorm / S_min_norm
        
        # S_nr (non-reciprocal component) - full-pol only
        S_nr = np.divide(delta, np.abs(delta), out=np.zeros_like(delta), where=np.abs(delta) > 1e-10)
        
        # Theta_rec (reciprocity angle) - full-pol only
        S_rec_norm_for_theta = np.sqrt(np.abs(S_rec1)**2 + np.abs(S_rec2)**2 + 
                                      np.abs(S_rec3)**2 + np.abs(S_rec4)**2)
        Theta_rec = np.arccos(np.clip(S_rec_norm_for_theta / np.maximum(a, 1e-10), 0, 1))
        
    else:
        # 3-pol case: [HH, HV, VV]
        S_hh = SAR_img[0]
        S_hv = SAR_img[1]  
        S_vv = SAR_img[2]
        
        # Compute norm of scattering vector (3-pol)
        a = np.sqrt(np.abs(S_hh)**2 + 2*np.abs(S_hv)**2 + np.abs(S_vv)**2)
        
        # Determine Pauli parameters (3-pol)
        alpha = (S_hh + S_vv) / np.sqrt(2)
        beta = (S_hh - S_vv) / np.sqrt(2)
        gamma = 2 * S_hv / np.sqrt(2)
        
        # Determine parameter x
        numerator = beta * np.conj(gamma) + np.conj(beta) * gamma
        denominator_sq = numerator**2 + (np.abs(beta)**2 - np.abs(gamma)**2)**2
        
        # Avoid division by zero
        valid_mask = denominator_sq > 1e-10
        sin_x = np.zeros_like(numerator, dtype=complex)
        cos_x = np.zeros_like(numerator, dtype=complex)
        
        sin_x[valid_mask] = numerator[valid_mask] / np.sqrt(denominator_sq[valid_mask])
        cos_x[valid_mask] = (np.abs(beta[valid_mask])**2 - np.abs(gamma[valid_mask])**2) / np.sqrt(denominator_sq[valid_mask])
        
        # Compute angle x
        x = np.zeros_like(sin_x, dtype=float)
        sin_x_real = np.real(sin_x)
        cos_x_real = np.real(cos_x)
        
        # Handle different quadrants
        mask1 = (sin_x_real >= 0)
        mask2 = (sin_x_real < 0) & (cos_x_real >= 0)
        mask3 = (sin_x_real < 0) & (cos_x_real < 0)
        
        x[mask1] = np.arccos(np.clip(cos_x_real[mask1], -1, 1))
        x[mask2] = np.arcsin(np.clip(sin_x_real[mask2], -1, 1))
        x[mask3] = -np.arcsin(np.clip(sin_x_real[mask3], -1, 1)) - np.pi
        
        # Determine DS (dominant symmetric component) - 3-pol
        scalar = (S_hh * np.cos(x/2) + 2*S_hv * np.sin(x/2) - S_vv * np.cos(x/2)) / np.sqrt(2)
        
        DS_1 = (alpha + np.cos(x/2) * scalar) / np.sqrt(2)
        DS_2 = np.sin(x/2) * scalar / np.sqrt(2)
        DS_3 = np.sin(x/2) * scalar / np.sqrt(2)
        DS_4 = (alpha - np.cos(x/2) * scalar) / np.sqrt(2)
        
        # Normalize S_max
        S_max_norm = np.sqrt(np.abs(DS_1)**2 + np.abs(DS_2)**2 + np.abs(DS_3)**2 + np.abs(DS_4)**2)
        S_max_norm = np.maximum(S_max_norm, 1e-10)
        
        S_max1 = DS_1 / S_max_norm
        S_max2 = DS_2 / S_max_norm
        S_max3 = DS_3 / S_max_norm  # Same as S_max2 for 3-pol
        S_max4 = DS_4 / S_max_norm
        
        # Determine S_rec (reciprocal component) - 3-pol
        S_rec1 = S_hh
        S_rec2 = S_hv
        S_rec3 = S_hv  # Same as S_rec2 for 3-pol
        S_rec4 = S_vv
        
        # Determine DS_rec
        scalar_rec = (S_rec1 * np.cos(x/2) + S_rec2 * np.sin(x/2) + 
                      S_rec3 * np.sin(x/2) - S_rec4 * np.cos(x/2)) / np.sqrt(2)
        
        DS_rec1 = (alpha + np.cos(x/2) * scalar_rec) / np.sqrt(2)
        DS_rec2 = np.sin(x/2) * scalar_rec / np.sqrt(2)
        DS_rec3 = np.sin(x/2) * scalar_rec / np.sqrt(2)
        DS_rec4 = (alpha - np.cos(x/2) * scalar_rec) / np.sqrt(2)
        
        # Determine S_min
        S_min1_unnorm = S_rec1 - DS_rec1
        S_min2_unnorm = S_rec2 - DS_rec2
        S_min3_unnorm = S_rec3 - DS_rec3
        S_min4_unnorm = S_rec4 - DS_rec4
        
        S_min_norm = np.sqrt(np.abs(S_min1_unnorm)**2 + np.abs(S_min2_unnorm)**2 + 
                            np.abs(S_min3_unnorm)**2 + np.abs(S_min4_unnorm)**2)
        S_min_norm = np.maximum(S_min_norm, 1e-10)
        
        S_min1 = S_min1_unnorm / S_min_norm
        S_min2 = S_min2_unnorm / S_min_norm
        S_min3 = S_min3_unnorm / S_min_norm
        S_min4 = S_min4_unnorm / S_min_norm
        
        # For 3-pol data, S_nr and Theta_rec are zero (reciprocal case)
        S_nr = np.zeros_like(a)
        Theta_rec = np.zeros_like(a)
    
    # Common processing for both cases
    # Determine Tau (separation angle)
    scalar_tau = (S_rec1 * np.conj(DS_1) + S_rec2 * np.conj(DS_2) + 
                  S_rec3 * np.conj(DS_3) + S_rec4 * np.conj(DS_4))
    
    S_rec_norm = np.sqrt(np.abs(S_rec1)**2 + np.abs(S_rec2)**2 + 
                        np.abs(S_rec3)**2 + np.abs(S_rec4)**2)
    DS_norm = np.sqrt(np.abs(DS_1)**2 + np.abs(DS_2)**2 + 
                     np.abs(DS_3)**2 + np.abs(DS_4)**2)
    
    # Avoid division by zero
    denom = S_rec_norm * DS_norm
    tau_arg = np.abs(scalar_tau) / np.maximum(denom, 1e-10)
    tau_arg = np.clip(tau_arg, 0, 1)
    Tau = np.arccos(tau_arg)
    
    # Determine Psi_D (diagonalization angle)
    Psi_1 = -x / 4
    
    # Test three candidate angles
    Psi_candidates = [Psi_1, Psi_1 + np.pi/2, Psi_1 - np.pi/2]
    
    # Initialize Psi_D
    Psi_D = np.zeros_like(Psi_1)
    
    for Psi_cand in Psi_candidates:
        # Check if angle is in valid range [-π/2, π/2]
        valid_range = (Psi_cand > -np.pi/2) & (Psi_cand <= np.pi/2)
        
        if np.any(valid_range):
            # Compute A1_1 and A1_4 for this candidate
            A1_1 = ((np.cos(Psi_cand)**2) * DS_rec1 - 
                    (np.cos(Psi_cand) * np.sin(Psi_cand)) * DS_rec2 -
                    (np.cos(Psi_cand) * np.sin(Psi_cand)) * DS_rec3 + 
                    (np.sin(Psi_cand)**2) * DS_rec4)
            
            A1_4 = ((np.sin(Psi_cand)**2) * DS_rec1 + 
                    (np.cos(Psi_cand) * np.sin(Psi_cand)) * DS_rec2 +
                    (np.cos(Psi_cand) * np.sin(Psi_cand)) * DS_rec3 + 
                    (np.cos(Psi_cand)**2) * DS_rec4)
            
            # Select where A1_1 >= A1_4 and angle is in valid range and Psi_D not set
            select_mask = valid_range & (np.abs(A1_1) >= np.abs(A1_4)) & (Psi_D == 0)
            Psi_D[select_mask] = Psi_cand[select_mask]
    
    # Apply final Psi_D correction based on diagonal condition
    A1_1_final = ((np.cos(Psi_D)**2) * DS_rec1 - 
                  (np.cos(Psi_D) * np.sin(Psi_D)) * DS_rec2 -
                  (np.cos(Psi_D) * np.sin(Psi_D)) * DS_rec3 + 
                  (np.sin(Psi_D)**2) * DS_rec4)
    
    A1_4_final = ((np.sin(Psi_D)**2) * DS_rec1 + 
                  (np.cos(Psi_D) * np.sin(Psi_D)) * DS_rec2 +
                  (np.cos(Psi_D) * np.sin(Psi_D)) * DS_rec3 + 
                  (np.cos(Psi_D)**2) * DS_rec4)
    
    # Additional Psi_D adjustments based on diagonal conditions
    I_a = (A1_1_final == A1_4_final)
    I_b = (A1_1_final == -A1_4_final)
    
    mask1 = (Psi_D > np.pi/4) & (I_a | I_b)
    mask2 = (Psi_D > -np.pi/4) & (Psi_D <= np.pi/4) & (~mask1) & (I_a | I_b)
    mask3 = (Psi_D <= -np.pi/4) & (~mask1) & (~mask2) & (I_a | I_b)
    
    Psi_D[mask1] = Psi_D[mask1] - np.pi/2
    # mask2 keeps original Psi_D
    Psi_D[mask3] = Psi_D[mask3] + np.pi/2
    
    # Always return 13 parameters for test compatibility
    return [S_max1, S_max2, S_max3, S_max4, S_min1, S_min2, S_min3, S_min4, S_nr, a, Tau, Theta_rec, Psi_D]


def cameron_classification(
    S_max1: np.ndarray,
    S_max2: np.ndarray,
    S_max3: np.ndarray,
    S_max4: np.ndarray,
    S_min1: np.ndarray,
    S_min2: np.ndarray,
    S_min3: np.ndarray,
    S_min4: np.ndarray,
    S_nr: np.ndarray,
    a: np.ndarray,
    Tau: np.ndarray,
    Theta_rec: np.ndarray,
    Psi_D: np.ndarray,
    method: int = 1
) -> np.ndarray:
    """
    Optimized Cameron classification (method 1 vectorized, fallback for method 2).

    Args:
        S_max1..S_max4, S_min1..S_min4, S_nr, a, Tau, Theta_rec, Psi_D: Parameter arrays from cameron_transform
        method: 1 for scalar-product, 2 for distance-based

    Returns:
        classification map (H, W) with integer labels 1-11
    """
    if method == 1:
        # Precompute trigonometric terms
        cos_tr = np.cos(Theta_rec)
        sin_tr = np.sin(Theta_rec)
        cos_ta = np.cos(Tau)
        sin_ta = np.sin(Tau)

        # Compute scattering matrix elements S1..S4
        s1 = a * cos_tr * (cos_ta * S_max1 + sin_ta * S_min1)
        s2 = a * cos_tr * (cos_ta * S_max2 + sin_ta * S_min2) - (a * sin_tr * S_nr / np.sqrt(2))
        s3 = a * cos_tr * (cos_ta * S_max3 + sin_ta * S_min3) + (a * sin_tr * S_nr / np.sqrt(2))
        s4 = a * cos_tr * (cos_ta * S_max4 + sin_ta * S_min4)

        H, W = a.shape
        cls = np.zeros((H, W), dtype=int)

        # 1) Non-reciprocal
        nr = Theta_rec > np.pi/4
        cls[nr] = 1

        # 2) Asymmetric & helix
        hel = (~nr) & (Tau > np.pi/8)
        left  = 0.5 * (s1 - s4 - 1j * (s2 + s3))
        right = 0.5 * (s1 - s4 + 1j * (s2 + s3))
        th_l = np.arccos(np.clip(np.abs(left)  / a, 0, 1))
        th_r = np.arccos(np.clip(np.abs(right) / a, 0, 1))

        asym = hel & (th_l > np.pi/4) & (th_r > np.pi/4)
        cls[asym] = 2

        lh = hel & ~asym & (th_l >= th_r)
        cls[lh] = 3
        rh = hel & ~asym & (th_l <  th_r)
        cls[rh] = 4

        # 3) Symmetric vs canonical targets
        sym = (~nr) & (Tau <= np.pi/8)
        # compute canonical scalars
        tri = (s1 + s4) / np.sqrt(2)
        dihedral = (s1 - s4) / np.sqrt(2)
        dipole   = s1
        cylinder = (2*s1 + s4) / np.sqrt(5)
        narrow   = (2*s1 - s4) / np.sqrt(5)
        quarter  = (s1 - 1j*s4) / np.sqrt(2)

        # angles to each target
        angles = np.stack([
            np.arccos(np.clip(np.abs(tri)      / a, 0, 1)),
            np.arccos(np.clip(np.abs(dihedral) / a, 0, 1)),
            np.arccos(np.clip(np.abs(dipole)   / a, 0, 1)),
            np.arccos(np.clip(np.abs(cylinder) / a, 0, 1)),
            np.arccos(np.clip(np.abs(narrow)   / a, 0, 1)),
            np.arccos(np.clip(np.abs(quarter)  / a, 0, 1)),
        ], axis=0)  # shape (6, H, W)

        min_ang = angles.min(axis=0)
        # Symmetric
        cls[sym & (min_ang > np.pi/4)] = 5

        # Assign canonical classes 6-11
        idx = angles.argmin(axis=0)
        for k in range(6):
            mask_k = sym & (idx == k) & (min_ang <= np.pi/4)
            cls[mask_k] = 6 + k

        return cls

    elif method == 2:
        # Fallback to existing implementation for method 2
        return cameron_classification_original(
            S_max1, S_max2, S_max3, S_max4,
            S_min1, S_min2, S_min3, S_min4,
            S_nr, a, Tau, Theta_rec, Psi_D,
            method=2
        )
    else:
        raise ValueError(f"Invalid method {method}: use 1 or 2.")


def cameron(
    SAR_img: np.ndarray,
    method: int = 1
) -> np.ndarray:
    """
    Full Cameron pipeline: compute parameters then classify.

    Args:
        SAR_img: Complex SAR image, shape (3,H,W) or (4,H,W)
        method: 1 or 2

    Returns:
        Classification map of shape (H,W) with labels 1-11
    """
    # validate input
    if SAR_img.ndim != 3 or SAR_img.shape[0] not in (3,4):
        raise ValueError(f"Expected SAR_img shape (3,H,W) or (4,H,W), got {SAR_img.shape}")

    # compute decomposition parameters
    params = cameron_transform(SAR_img)
    # classify
    return cameron_classification(*params, method=method)


def compute_cameron_metrics(
    cameron_original: np.ndarray,
    cameron_generated: np.ndarray,
    class_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for Cameron classification comparison.

    Args:
        cameron_original: Original Cameron classification map
        cameron_generated: Generated/reconstructed Cameron classification map  
        class_labels: List of class labels (defaults to Cameron classes [1-11])

    Returns:
        Dictionary containing various classification metrics
        
    Raises:
        ValueError: If input arrays don't have the same shape
    """
    if cameron_original.shape != cameron_generated.shape:
        raise ValueError(f"Input arrays must have the same shape. "
                        f"Got {cameron_original.shape} and {cameron_generated.shape}")
    
    if class_labels is None:
        class_labels = list(range(1, 12))  # Cameron classes 1-11

    # Flatten arrays for sklearn metrics
    y_true = cameron_original.flatten()
    y_pred = cameron_generated.flatten()

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix (normalized and raw)
    cm_raw = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_normalized = confusion_matrix(
        y_true, y_pred, labels=class_labels, normalize="true"
    )

    # Per-class metrics
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )

    # Macro and micro averages
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )
    precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, fscore_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=class_labels, average="weighted", zero_division=0
        )
    )

    # Additional metrics
    kappa = cohen_kappa_score(y_true, y_pred)

    # Try to compute Matthews correlation coefficient
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        # MCC not defined for some edge cases
        mcc = np.nan

    # Class distribution in original and generated
    unique_orig, counts_orig = np.unique(y_true, return_counts=True)
    unique_gen, counts_gen = np.unique(y_pred, return_counts=True)

    class_distribution_orig = {
        int(cls): int(count) for cls, count in zip(unique_orig, counts_orig)
    }
    class_distribution_gen = {
        int(cls): int(count) for cls, count in zip(unique_gen, counts_gen)
    }

    # Create comprehensive metrics dictionary
    metrics = {
        # Overall metrics
        "accuracy": float(accuracy),
        "cohen_kappa": float(kappa),
        "matthews_corrcoef": float(mcc) if not np.isnan(mcc) else None,
        # Macro averages
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(fscore_macro),
        # Micro averages
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(fscore_micro),
        # Weighted averages
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(fscore_weighted),
        # Per-class metrics
        "per_class_metrics": {
            int(class_labels[i]): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(fscore[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_labels))
        },
        # Class distributions
        "class_distribution_original": class_distribution_orig,
        "class_distribution_generated": class_distribution_gen,
        # Confusion matrices
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "class_labels": class_labels,
    }

    return metrics


def plot_dataset_split_mask(
    mask: np.ndarray,
    patch_size: int,
    train_indices: List[int],
    valid_indices: List[int],
    test_indices: Optional[List[int]] = None,
    savepath: Optional[Union[str, Path]] = None,
    title: str = "Dataset Split Visualization",
    wandb_log: bool = False,
) -> None:
    """
    Create a spatial visualization of how the dataset is split into train/validation/test sets.

    This function creates a mask showing which patches belong to which set, mapped back to
    their spatial coordinates in the original image.

    Args:
        cfg: Configuration dictionary containing crop and patch information
        train_indices: List of patch indices used for training
        valid_indices: List of patch indices used for validation
        test_indices: List of patch indices used for testing (optional)
        savepath: Path to save the plot (optional)
        title: Title for the plot
        wandb_log: Whether to log to wandb

    Raises:
        ValueError: If crop information is not available in config
    """
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

    # Define colors for each set
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red
    labels = ["Train", "Validation", "Test", "Unassigned"]

    # Create custom colormap
    n_sets = 3 if test_indices is not None else 2
    # Include unassigned (-1) in colormap
    bounds = list(range(-1, n_sets + 1))
    cmap = ListedColormap([colors[3]] + colors[:n_sets])  # unassigned + actual sets
    norm = BoundaryNorm(bounds, len(bounds) - 1)

    # Plot the mask
    im = ax.imshow(mask, cmap=cmap, norm=norm, origin="lower", aspect="equal")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, boundaries=bounds, ticks=bounds[:-1])
    cbar_labels = [labels[3]] + labels[:n_sets]  # unassigned + actual sets
    cbar.set_ticklabels(cbar_labels)

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column Index", fontsize=12)
    ax.set_ylabel("Row Index", fontsize=12)

    # Add grid for better visualization of patches
    # Draw patch boundaries
    for h in range(0, mask.shape[0], patch_size):
        ax.axhline(y=h, color="white", linewidth=0.5, alpha=0.3)
    for w in range(0, mask.shape[1], patch_size):
        ax.axvline(x=w, color="white", linewidth=0.5, alpha=0.3)

    # Add statistics text
    stats_text = f"Train patches: {len(train_indices)}\n"
    stats_text += f"Validation patches: {len(valid_indices)}\n"
    if test_indices is not None:
        stats_text += f"Test patches: {len(test_indices)}\n"
    stats_text += f"Patch size: {patch_size}×{patch_size}\n"
    stats_text += f"Patch stride: {patch_size}\n"
    stats_text += f"Image size: {mask.shape[0]}×{mask.shape[1]}"

    # Add text box with statistics
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Save plot if path provided
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        logger.info(f"Dataset split visualization saved to {savepath}")

    # Log to wandb if enabled
    if wandb_log and wandb.run:
        wandb.log(
            {
                "dataset_split/visualization": wandb.Image(fig, caption=title),
                "dataset_split/train_patches": len(train_indices),
                "dataset_split/valid_patches": len(valid_indices),
                "dataset_split/test_patches": len(test_indices) if test_indices else 0,
                "dataset_split/patch_size": patch_size,
                "dataset_split/total_patches": len(train_indices)
                + len(valid_indices)
                + (len(test_indices) if test_indices else 0),
            }
        )

    plt.close(fig)

    logger.info(
        f"Dataset split visualization created with {len(train_indices)} train, {len(valid_indices)} validation"
        + (f", {len(test_indices)} test patches" if test_indices else " patches")
    )


def create_dataset_split_visualization(
    cfg: Dict[str, Any],
    logdir: Union[str, Path],
    mask: np.ndarray,
    train_indices: List[int],
    valid_indices: List[int],
    test_indices: Optional[List[int]] = None,
    wandb_log: bool = False,
) -> Optional[str]:
    """
    Convenience function to create dataset split visualization for experiments.

    This function automatically generates the dataset split indices and creates
    a visualization, making it easy to integrate into experiment workflows.

    Args:
        cfg: Configuration dictionary from experiment
        logdir: Directory to save the visualization
        wandb_log: Whether to log to wandb

    Returns:
        Path to saved visualization file, or None if an error occurred

    Example:
        >>> # In your experiment class
        >>> viz_path = create_dataset_split_visualization(
        ...     cfg=self.cfg,
        ...     logdir=self.logdir,
        ...     wandb_log=True
        ... )
        >>> if viz_path:
        ...     logger.info(f"Dataset split visualization saved to {viz_path}")
    """

    # Create output path
    dataset_name = cfg["data"]["dataset"]["name"]
    filename = f"dataset_split_{dataset_name.lower()}.png"
    output_path = Path(logdir) / filename

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the visualization
    plot_dataset_split_mask(
        mask=mask,
        patch_size=cfg["data"]["patch_size"],
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
        savepath=str(output_path),
        title=f"{dataset_name} Dataset Split",
        wandb_log=wandb_log,
    )


def visualize_segmentation_full_image(
    model: torch.nn.Module,
    ground_truth_segmentation: np.ndarray,
    predicted_segmentation: np.ndarray,
    original_image: np.ndarray,
    cfg: Dict[str, Any],
    logdir: Union[str, Path],
    number_classes: int,
    device: torch.device = None,
    wandb_log: bool = False,
) -> None:
    """
    Create side-by-side visualization of full image segmentation: Original Image | Ground Truth | Prediction.

    Args:
        model: Trained segmentation model
        ground_truth_segmentation: Full ground truth segmentation image
        predicted_segmentation: Full predicted segmentation image
        original_image: Original SAR image (complex-valued, shape: [3, height, width])
        cfg: Configuration dictionary
        logdir: Directory to save visualization
        number_classes: Number of segmentation classes
        device: Device for inference
        wandb_log: Whether to log to wandb
    """

    logger.info("Creating full image segmentation visualization...")

    # Get device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            logger.warning("Model has no parameters; defaulting to CPU")
            device = torch.device("cpu")

    model.to(device).eval()

    # Check dataset type to determine visualization approach
    dataset_type = get_dataset_type_from_config(cfg)

    # Prepare original image for visualization
    if dataset_type == "polsar":
        # Use Pauli decomposition for PolSAR data
        img_original = exp_amplitude_transform(original_image).numpy()
        pauli_img_original = pauli_transform(img_original).transpose(1, 2, 0)
        eq_original, _ = equalize(pauli_img_original)
        original_title = "Original Image - Pauli Decomposition"
    else:
        # For non-PolSAR datasets, use simple intensity/amplitude display
        if np.iscomplexobj(original_image):
            # Show amplitude for complex data
            img_display = np.abs(original_image)
            if img_display.ndim == 3 and img_display.shape[0] > 1:
                img_display = img_display[0]  # Take first channel for display
        else:
            # Show as-is for real data
            img_display = original_image
            if img_display.ndim == 3 and img_display.shape[0] > 1:
                img_display = img_display[0]  # Take first channel for display

        eq_original, _ = equalize(img_display)
        original_title = "Original Image"

    # Set up segmentation colormap
    class_colors = {
        7: {
            0: "black",
            1: "purple",
            2: "blue",
            3: "green",
            4: "red",
            5: "cyan",
            6: "yellow",
        },
        5: {
                        0: "black",
            1: "green",
            2: "brown",
            3: "blue",
            4: "yellow",
        },
    }.get(number_classes, {})

    cmap = ListedColormap([class_colors[key] for key in sorted(class_colors.keys())])
    bounds = np.arange(len(class_colors) + 1) - 0.5
    norm = BoundaryNorm(bounds, len(class_colors))
    patches = [
        mpatches.Patch(color=class_colors[i], label=f"Class {i}")
        for i in sorted(class_colors.keys())
    ]

    # Mask prediction if ignore_index is provided
    ignore_index = cfg["training"].get("ignore_index", None)
    if ignore_index is not None:
        masked_predicted = predicted_segmentation.copy()
        masked_predicted[ground_truth_segmentation == ignore_index] = ignore_index
    else:
        masked_predicted = predicted_segmentation

    # Create four-panel visualization
    _, axes = plt.subplots(1, 4, figsize=(24, 8), constrained_layout=True)

    # Original Image (adaptive based on dataset type)
    axes[0].imshow(eq_original, origin="lower")
    axes[0].set_title(original_title, fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Ground Truth
    axes[1].imshow(ground_truth_segmentation, cmap=cmap, norm=norm, origin="lower")
    axes[1].set_title("Ground Truth Segmentation", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(predicted_segmentation, cmap=cmap, norm=norm, origin="lower")
    axes[2].set_title("Predicted Segmentation", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(masked_predicted, cmap=cmap, norm=norm, origin="lower")
    axes[3].set_title(
        "Masked Prediction (Ignore Index)", fontsize=14, fontweight="bold"
    )
    axes[3].axis("off")

    # Add legend to the figure
    plt.figlegend(
        handles=patches, loc="center right", title="Classes", bbox_to_anchor=(1.02, 0.5)
    )

    plt.suptitle("Full Image Segmentation Analysis", fontsize=16, fontweight="bold")

    # Save visualization
    save_path = Path(logdir) / "segmentation_full_image.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Full image segmentation visualization saved to {save_path}")

    # Log to wandb if enabled
    if wandb_log and hasattr(wandb, "run") and wandb.run:
        wandb.log(
            {
                "segmentation_full_image": wandb.Image(
                    str(save_path), caption="Full Image Segmentation Analysis"
                )
            }
        )
