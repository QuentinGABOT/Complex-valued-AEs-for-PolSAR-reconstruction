import torch
import numpy as np
import wandb
from typing import List, Tuple, Dict, Any, Optional
from cvnn.utils import setup_logging
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

# module-level logger
logger = setup_logging(__name__)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def complex_ssim(
    img1_complex,
    img2_complex,
    window_size=11,
    size_average=True,
    magnitude_weight=0.7,
    phase_weight=0.3,
    magnitude_threshold=1e-6,
):
    """
    Compute SSIM for complex-valued images with improved handling of magnitude and phase.

    Args:
        img1_complex, img2_complex: Complex-valued tensors
        window_size: SSIM window size
        size_average: Whether to average across spatial dimensions
        magnitude_weight: Weight for magnitude SSIM (default 0.7)
        phase_weight: Weight for phase SSIM (default 0.3)
        magnitude_threshold: Threshold below which phase is ignored

    Returns:
        Complex SSIM score
    """
    # Extract magnitude and phase
    mag1 = torch.abs(img1_complex)
    mag2 = torch.abs(img2_complex)
    phase1 = torch.angle(img1_complex)
    phase2 = torch.angle(img2_complex)

    # Compute magnitude SSIM
    mag_ssim = ssim(mag1, mag2, window_size, size_average)

    # Handle phase wrapping by computing the wrapped phase difference
    phase_diff = torch.angle(torch.exp(1j * (phase1 - phase2)))

    # Create magnitude-weighted phase similarity
    # Only consider phase where both magnitudes are above threshold
    mag_mask = (mag1 > magnitude_threshold) & (mag2 > magnitude_threshold)

    if mag_mask.any():
        # Compute phase similarity using cosine of phase difference
        phase_similarity = torch.cos(phase_diff)

        # Weight phase similarity by the geometric mean of magnitudes
        magnitude_weights = torch.sqrt(mag1 * mag2)
        weighted_phase_sim = phase_similarity * magnitude_weights * mag_mask.float()

        # Normalize by the sum of weights
        total_weights = magnitude_weights * mag_mask.float()
        if total_weights.sum() > 0:
            phase_sim_score = weighted_phase_sim.sum() / total_weights.sum()
        else:
            phase_sim_score = torch.tensor(1.0, device=img1_complex.device)
    else:
        # If no valid magnitude pixels, ignore phase
        phase_sim_score = torch.tensor(1.0, device=img1_complex.device)

    # Combine magnitude and phase SSIM
    complex_ssim_score = magnitude_weight * mag_ssim + phase_weight * phase_sim_score

    return complex_ssim_score


def complex_ssim_correlation(
    img1_complex, img2_complex, window_size=11, size_average=True, alpha=0.8, beta=0.2
):
    """
    Alternative complex SSIM using complex correlation approach.

    This method computes SSIM directly on complex values using complex correlation
    and combines it with magnitude-based SSIM.

    Args:
        img1_complex, img2_complex: Complex-valued tensors
        window_size: SSIM window size
        size_average: Whether to average across spatial dimensions
        alpha: Weight for magnitude SSIM
        beta: Weight for complex correlation term

    Returns:
        Complex SSIM score
    """
    # Magnitude SSIM
    mag1 = torch.abs(img1_complex)
    mag2 = torch.abs(img2_complex)
    mag_ssim = ssim(mag1, mag2, window_size, size_average)

    # Complex correlation coefficient
    # Normalize complex images
    img1_norm = img1_complex / (torch.abs(img1_complex) + 1e-8)
    img2_norm = img2_complex / (torch.abs(img2_complex) + 1e-8)

    # Compute complex correlation
    correlation = torch.mean(torch.real(img1_norm * torch.conj(img2_norm)))

    # Ensure correlation is in [0, 1] range
    correlation = (correlation + 1) / 2

    # Combine terms
    complex_ssim_score = alpha * mag_ssim + beta * correlation

    return complex_ssim_score


def complex_ssim_advanced(
    img1_complex, img2_complex, window_size=11, size_average=True
):
    """
    Advanced complex SSIM that considers both local structure and global phase coherence.

    This implementation:
    1. Uses magnitude SSIM for structural information
    2. Uses local phase coherence for phase information
    3. Applies perceptual weighting based on magnitude

    Args:
        img1_complex, img2_complex: Complex-valued tensors
        window_size: SSIM window size
        size_average: Whether to average across spatial dimensions

    Returns:
        Advanced complex SSIM score
    """
    # Extract components
    mag1 = torch.abs(img1_complex)
    mag2 = torch.abs(img2_complex)

    # Magnitude SSIM (primary structural measure)
    mag_ssim = ssim(mag1, mag2, window_size, size_average)

    # Check if image is large enough for sliding window approach
    B, C, H, W = img1_complex.shape
    if H < window_size or W < window_size:
        # For small images, fall back to simpler correlation-based approach
        return complex_ssim_correlation(
            img1_complex, img2_complex, min(window_size, min(H, W)), size_average
        )

    # Local phase coherence using complex correlation in sliding windows
    pad = window_size // 2

    # Pad images
    img1_padded = F.pad(img1_complex, (pad, pad, pad, pad), mode="reflect")
    img2_padded = F.pad(img2_complex, (pad, pad, pad, pad), mode="reflect")

    phase_coherence_map = torch.zeros_like(mag1)

    # Compute local phase coherence
    for i in range(H):
        for j in range(W):
            # Extract local windows
            window1 = img1_padded[:, :, i : i + window_size, j : j + window_size]
            window2 = img2_padded[:, :, i : i + window_size, j : j + window_size]

            # Normalize by magnitude to focus on phase
            norm1 = window1 / (torch.abs(window1) + 1e-8)
            norm2 = window2 / (torch.abs(window2) + 1e-8)

            # Local phase correlation
            correlation = torch.mean(torch.real(norm1 * torch.conj(norm2)), dim=(2, 3))
            phase_coherence_map[:, :, i, j] = (correlation + 1) / 2

    # Weight phase coherence by magnitude importance
    magnitude_weights = torch.sqrt(mag1 * mag2)
    weighted_phase_coherence = torch.sum(
        phase_coherence_map * magnitude_weights
    ) / torch.sum(magnitude_weights)

    # Perceptual weighting: magnitude is more important than phase
    # but phase becomes more important at higher magnitudes
    magnitude_mean = torch.mean(magnitude_weights)
    phase_weight = torch.clamp(magnitude_mean, 0.1, 0.4)  # Adaptive weighting
    magnitude_weight = 1 - phase_weight

    # Final complex SSIM
    complex_ssim_score = (
        magnitude_weight * mag_ssim + phase_weight * weighted_phase_coherence
    )

    return complex_ssim_score


def evaluate_reconstruction(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    cfg: dict = None,
    device: torch.device = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Main evaluation dispatcher that automatically selects the appropriate evaluation method
    based on dataset type and configuration.

    Args:
        test_loader: DataLoader for test data
        model: Model to evaluate
        cfg: Configuration dictionary containing dataset and model info (optional)
        device: Device to run evaluation on
        **kwargs: Additional arguments passed to specific evaluation functions

    Returns:
        Dictionary with reconstruction metrics
    """
    # If no config provided, use default complex evaluation
    if cfg is None:
        return evaluate_reconstruction_complex_polsar(
            test_loader, model, device, **kwargs
        )

    from cvnn.data import get_dataset_info, validate_and_correct_config

    # Validate and correct config if needed
    cfg = validate_and_correct_config(cfg.copy())

    # Check if dataset info is available after validation
    if (
        "data" not in cfg
        or "dataset" not in cfg["data"]
        or "name" not in cfg["data"]["dataset"]
    ):
        return evaluate_reconstruction_complex_polsar(
            test_loader, model, device, **kwargs
        )

    if "model" not in cfg or "layer_mode" not in cfg["model"]:
        return evaluate_reconstruction_complex_polsar(
            test_loader, model, device, **kwargs
        )

    dataset_name = cfg["data"]["dataset"]["name"]
    dataset_info = get_dataset_info(dataset_name)
    layer_mode = cfg["model"]["layer_mode"]
    real_pipeline_type = cfg["data"].get("real_pipeline_type")

    logger.info(
        f"Evaluating reconstruction for {dataset_info['type']} dataset '{dataset_name}'"
    )
    logger.info(f"Layer mode: {layer_mode}, Real pipeline: {real_pipeline_type}")

    # Dispatch to appropriate evaluation function
    if layer_mode == "real" and real_pipeline_type:
        if real_pipeline_type == "complex_dual_real":
            return evaluate_reconstruction_dual_real(
                test_loader, model, device, **kwargs
            )
        elif real_pipeline_type == "complex_amplitude_real":
            return evaluate_reconstruction_amplitude(
                test_loader, model, device, **kwargs
            )
        elif real_pipeline_type == "real_real":
            return evaluate_reconstruction_real(test_loader, model, device, **kwargs)

    # Default: complex PolSAR evaluation
    return evaluate_reconstruction_complex_polsar(test_loader, model, device, **kwargs)


def evaluate_reconstruction_dual_real(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = None,
    **kwargs,
) -> Dict[str, float]:
    """Evaluate reconstruction for complex_dual_real pipeline.

    Expects test_loader to provide dual format: (original_complex, transformed_dual_real)
    - Uses transformed_dual_real for model input
    - Reconstructs complex from model output
    - Compares with original_complex ground truth
    """
    logger.info("Evaluating reconstruction with dual real to complex conversion")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    mse = 0.0
    psnr = 0.0
    ssim_score = 0.0
    shift_consistency_total = 0.0

    for batch in test_loader:
        # Extract inputs from batch
        dual_real_input = batch[0] if isinstance(batch, (tuple, list)) else batch
        dual_real_input = dual_real_input.to(device)

        # Convert dual real input back to complex for ground truth
        if dual_real_input.shape[1] % 2 != 0:
            raise ValueError(
                f"Expected even number of channels for dual real input, got {dual_real_input.shape[1]}"
            )

        num_complex_channels = dual_real_input.shape[1] // 2
        real_part_input = dual_real_input[:, :num_complex_channels]
        imag_part_input = dual_real_input[:, num_complex_channels:]
        original_complex = torch.complex(real_part_input, imag_part_input)

        with torch.no_grad():
            outputs = model(dual_real_input)  # Model processes dual real input

        # Compute circular shift consistency for this batch
        batch_consistency = circular_shift_consistency(model, dual_real_input, "reconstruction", device)
        shift_consistency_total += batch_consistency

        # Reconstruct complex tensor from dual real output
        num_channels = outputs.shape[1]
        if num_channels % 2 != 0:
            raise ValueError(
                f"Expected even number of channels for dual real output, got {num_channels}"
            )

        half_channels = num_channels // 2
        real_part = outputs[:, :half_channels, ...]
        imag_part = outputs[:, half_channels:, ...]
        complex_output = torch.complex(real_part, imag_part)

        # Compare with original complex ground truth
        original = original_complex.cpu()
        reconstructed = complex_output.cpu()

        # Use complex SSIM
        ssim_score += complex_ssim(original, reconstructed).item()

        # Convert to numpy for MSE and PSNR
        original_np = original.numpy()
        reconstructed_np = reconstructed.numpy()

        # Compute MSE
        batch_mse = np.mean(np.abs(original_np - reconstructed_np) ** 2)
        mse += float(batch_mse)

        # Compute PSNR
        max_val = np.max(np.abs(original_np))
        if batch_mse > 0:
            batch_psnr = 20 * np.log10(max_val / np.sqrt(batch_mse))
            psnr += float(batch_psnr)
        else:
            psnr += float("inf")

    # Compute averages
    avg_mse = mse / len(test_loader)
    avg_psnr = psnr / len(test_loader)
    avg_ssim = ssim_score / len(test_loader)
    avg_shift_consistency = shift_consistency_total / len(test_loader)
    
    metrics = {
        "mse": avg_mse,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "circular_shift_consistency": avg_shift_consistency,
    }

    logger.info(f"Dual Real -> Complex MSE: {avg_mse:.4f}")
    logger.info(f"Dual Real -> Complex PSNR: {avg_psnr:.4f}")
    logger.info(f"Dual Real -> Complex SSIM: {avg_ssim:.4f}")
    logger.info(f"Dual Real -> Complex Circular Shift Consistency: {avg_shift_consistency:.4f}")

    if wandb.run:
        wandb.log({f"reconstruction/{k}": v for k, v in metrics.items()})

    return metrics


def evaluate_reconstruction_amplitude(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = None,
    **kwargs,
) -> Dict[str, float]:
    """Evaluate reconstruction for complex_amplitude_real pipeline.

    Expects test_loader to provide dual format: (original_complex, transformed_amplitude)
    - Uses transformed_amplitude for model input
    - Compares model amplitude output with original complex data amplitude
    """
    logger.info("Evaluating amplitude reconstruction")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    mse = 0.0
    psnr = 0.0
    ssim_score = 0.0
    shift_consistency_total = 0.0

    for batch in test_loader:
        # Extract inputs from batch
        amplitude_input = batch[0] if isinstance(batch, (tuple, list)) else batch
        amplitude_input = amplitude_input.to(device)

        # For amplitude pipeline, we need to get the original complex data
        # This is a limitation - we can't perfectly reconstruct the original complex from amplitude
        # For evaluation purposes, we'll get the ground truth amplitude by assuming the input
        # was originally complex data that got converted to amplitude
        # Since we can't recover the phase, we'll just compare amplitudes

        with torch.no_grad():
            outputs = model(amplitude_input)  # Real amplitude output

        # Compute circular shift consistency for this batch
        batch_consistency = circular_shift_consistency(model, amplitude_input, "reconstruction", device)
        shift_consistency_total += batch_consistency

        # Compare amplitude predictions with amplitude inputs (as proxy for ground truth)
        original_amplitude = amplitude_input.cpu()
        reconstructed_amplitude = outputs.cpu()

        # Use real SSIM for amplitude comparison
        ssim_score += ssim(original_amplitude, reconstructed_amplitude).item()

        # Convert to numpy
        original_np = original_amplitude.numpy()
        reconstructed_np = reconstructed_amplitude.numpy()

        # Compute MSE
        batch_mse = np.mean((original_np - reconstructed_np) ** 2)
        mse += float(batch_mse)

        # Compute PSNR
        max_val = np.max(original_np)
        if batch_mse > 0:
            batch_psnr = 20 * np.log10(max_val / np.sqrt(batch_mse))
            psnr += float(batch_psnr)
        else:
            psnr += float("inf")

    # Compute averages
    avg_mse = mse / len(test_loader)
    avg_psnr = psnr / len(test_loader)
    avg_ssim = ssim_score / len(test_loader)
    avg_shift_consistency = shift_consistency_total / len(test_loader)

    metrics = {
        "mse": avg_mse,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "circular_shift_consistency": avg_shift_consistency,
    }

    logger.info(f"Amplitude MSE: {avg_mse:.4f}")
    logger.info(f"Amplitude PSNR: {avg_psnr:.4f}")
    logger.info(f"Amplitude SSIM: {avg_ssim:.4f}")
    logger.info(f"Amplitude Circular Shift Consistency: {avg_shift_consistency:.4f}")

    if wandb.run:
        wandb.log({f"reconstruction/{k}": v for k, v in metrics.items()})

    return metrics


def evaluate_reconstruction_real(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = None,
    **kwargs,
) -> Dict[str, float]:
    """Evaluate reconstruction for real_real pipeline.

    Standard real-valued image reconstruction metrics.
    """
    logger.info("Evaluating real-valued reconstruction")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    mse = 0.0
    psnr = 0.0
    ssim_score = 0.0
    shift_consistency_total = 0.0

    for batch in test_loader:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        # Compute circular shift consistency for this batch
        batch_consistency = circular_shift_consistency(model, inputs, "reconstruction", device)
        shift_consistency_total += batch_consistency

        original = inputs.cpu()
        reconstructed = outputs.cpu()

        # Use real SSIM
        ssim_score += ssim(original, reconstructed).item()

        # Convert to numpy
        original_np = original.numpy()
        reconstructed_np = reconstructed.numpy()

        # Compute MSE
        batch_mse = np.mean((original_np - reconstructed_np) ** 2)
        mse += float(batch_mse)

        # Compute PSNR
        max_val = np.max(original_np)
        if batch_mse > 0:
            batch_psnr = 20 * np.log10(max_val / np.sqrt(batch_mse))
            psnr += float(batch_psnr)
        else:
            psnr += float("inf")

    # Compute averages
    avg_mse = mse / len(test_loader)
    avg_psnr = psnr / len(test_loader)
    avg_ssim = ssim_score / len(test_loader)
    avg_shift_consistency = shift_consistency_total / len(test_loader)

    metrics = {
        "mse": avg_mse,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "circular_shift_consistency": avg_shift_consistency,
    }

    logger.info(f"Real MSE: {avg_mse:.4f}")
    logger.info(f"Real PSNR: {avg_psnr:.4f}")
    logger.info(f"Real SSIM: {avg_ssim:.4f}")
    logger.info(f"Real Circular Shift Consistency: {avg_shift_consistency:.4f}")

    if wandb.run:
        wandb.log({f"reconstruction/{k}": v for k, v in metrics.items()})

    return metrics


def evaluate_reconstruction_complex_polsar(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = None,
    detailed_ssim: bool = False,
    adaptive_ssim: bool = True,
) -> Dict[str, float]:
    """Original complex PolSAR evaluation function.

    This is the existing evaluate_reconstruction logic for complex PolSAR data.
    """
    # This is the existing logic from the original evaluate_reconstruction function
    # I'm keeping it separate for clarity
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            logger.warning("Model has no parameters; defaulting to CPU")
            device = torch.device("cpu")

    mse = 0.0
    psnr = 0.0
    ssim_score = 0.0
    ssim_detailed = {} if detailed_ssim else None
    ssim_methods_used = [] if adaptive_ssim else None

    for batch in test_loader:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        original = inputs.cpu()
        reconstructed = outputs.cpu()

        # Compute complex SSIM using improved implementation
        if adaptive_ssim:
            batch_ssim, method_used = adaptive_complex_ssim(original, reconstructed)
            ssim_score += batch_ssim
            ssim_methods_used.append(method_used)
        else:
            ssim_score += complex_ssim(original, reconstructed).item()

        # Compute detailed SSIM metrics if requested
        if detailed_ssim:
            batch_detailed = evaluate_complex_ssim_methods(original, reconstructed)
            if ssim_detailed is None or len(ssim_detailed) == 0:
                ssim_detailed = {
                    key: [] for key in batch_detailed.keys() if key != "analysis"
                }
                if "analysis" in batch_detailed:
                    ssim_detailed["analysis"] = {
                        k: [] for k in batch_detailed["analysis"].keys()
                    }

            for key, value in batch_detailed.items():
                if key != "analysis":
                    ssim_detailed[key].append(value)
                else:
                    for k, v in value.items():
                        ssim_detailed["analysis"][k].append(v)

        # Convert to numpy for MSE and PSNR calculations
        original = original.numpy()
        reconstructed = reconstructed.numpy()

        # compute MSE
        batch_mse = np.mean(np.abs(original - reconstructed) ** 2)
        mse += float(batch_mse)

        # compute PSNR (for complex data, use magnitude)
        max_val = np.max(np.abs(original))
        if batch_mse > 0:
            batch_psnr = 20 * np.log10(max_val / np.sqrt(batch_mse))
            psnr += float(batch_psnr)
        else:
            psnr += float("inf")

    # Compute averages
    avg_mse = mse / len(test_loader)
    avg_psnr = psnr / len(test_loader)
    avg_ssim = ssim_score / len(test_loader)

    if adaptive_ssim and ssim_methods_used:
        from collections import Counter

        method_counts = Counter(ssim_methods_used)
        print(f"SSIM methods used: {dict(method_counts)}")

    # Prepare metrics for logging and return
    metrics = {
        "mse": avg_mse,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
    }
    logger.info(f"Complex PolSAR MSE: {avg_mse:.4f}")
    logger.info(f"Complex PolSAR PSNR: {avg_psnr:.4f}")
    logger.info(f"Complex PolSAR SSIM: {avg_ssim:.4f}")

    # Add detailed SSIM metrics if computed
    if detailed_ssim and ssim_detailed:
        for key, values in ssim_detailed.items():
            if key == "analysis":
                for k, v in values.items():
                    metrics[f"ssim_analysis_{k}"] = np.mean(v)
            else:
                metrics[f"ssim_{key}"] = np.mean(values)

    # Compute circular shift consistency
    shift_consistencies = []
    for batch in test_loader:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        inputs = inputs.to(device)
        
        consistency = circular_shift_consistency(model, inputs, "reconstruction", device)
        shift_consistencies.append(consistency)
    
    avg_shift_consistency = np.mean(shift_consistencies)
    metrics["circular_shift_consistency"] = avg_shift_consistency
    logger.info(f"Complex PolSAR Circular Shift Consistency: {avg_shift_consistency:.4f}")

    if wandb.run:
        wandb_log_dict = {}
        for key, value in metrics.items():
            wandb_log_dict[f"reconstruction/{key}"] = value
        wandb.log(wandb_log_dict)

    return metrics


def reconstruct_full_image(
    model: torch.nn.Module,
    full_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct full image by reassembling patches from `full_loader`.
    Uses mini-batch processing to avoid BatchNorm issues with single samples.

    Args:
        model: The trained model for reconstruction
        full_loader: DataLoader containing image patches
        config: Configuration dictionary containing batch_size
        device: Device to run inference on

    Returns:
        tuple: (original_image, reconstructed_image), each with shape
               (num_channels, nb_rows, nb_cols).
    """
    # determine device, fallback if no parameters
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            logger.warning("Model has no parameters; defaulting to CPU device")
            device = torch.device("cpu")
    model.to(device).eval()

    # collect original patches and their indices together
    original_segments: List[np.ndarray] = []
    collected_original_indices: List[int] = []

    for data in full_loader:
        inputs = data[0] if isinstance(data, (tuple, list)) else data
        original_segments.extend([seg for seg in inputs.cpu().numpy()])

        # Extract indices from the same batch
        if len(data) >= 2:

            indices = data[1] if len(data) == 2 else data[2]
            collected_original_indices.extend(indices.cpu().detach().numpy())

    # Verify that segments and indices have the same length
    if len(collected_original_indices) == 0:
        # No indices provided by DataLoader, use sequential placement
        collected_original_indices = None
    elif len(original_segments) != len(collected_original_indices):
        logger.warning(
            f"Mismatch between segments ({len(original_segments)}) and indices ({len(collected_original_indices)}). "
            f"Using sequential placement."
        )
        collected_original_indices = None  # Fall back to sequential placement

    patch_size = config["data"]["patch_size"]

    # Use inferred_input_channels if available, fallback to num_channels for tests
    num_channels = (
        config["data"].get("inferred_input_channels") or config["data"]["num_channels"]
    )

    # Determine expected output channels based on pipeline type
    layer_mode = config.get("model", {}).get("layer_mode", "complex")
    real_pipeline_type = config.get("data", {}).get("real_pipeline_type")

    # Determine actual channels from the data for original image assembly
    actual_channels = (
        original_segments[0].shape[0] if original_segments else num_channels
    )

    # For real pipelines, we need to handle the data format correctly
    if layer_mode == "real" and real_pipeline_type:
        # The full_loader provides transformed data (e.g., 6 channels for dual real)
        # For original image assembly, we need the original format
        if real_pipeline_type == "complex_dual_real":
            # The input data is dual real (6 channels), convert segments back to complex for original image
            complex_original_segments = []
            for seg in original_segments:
                if seg.shape[0] % 2 != 0:
                    raise ValueError(
                        f"Dual real data must have even number of channels, got {seg.shape[0]}"
                    )
                C = seg.shape[0] // 2
                real_part = seg[:C]
                imag_part = seg[C:]
                complex_seg = real_part + 1j * imag_part
                complex_original_segments.append(complex_seg)

            # Assemble original image using complex data (original channels)
            # For complex_dual_real, the complex segments have half the channels of the original segments
            original_complex_channels = actual_channels // 2
            original_image = _assemble_image(
                complex_original_segments,
                original_complex_channels,  # Use original complex channels (3), not inferred channels (6)
                nsamples_per_rows,
                nsamples_per_cols,
                patch_size,
                collected_original_indices,
            )
        elif real_pipeline_type == "complex_amplitude_real":
            # For amplitude, create approximate complex from amplitude (zero phase)
            complex_original_segments = []
            for seg in original_segments:
                complex_seg = seg + 1j * np.zeros_like(seg)
                complex_original_segments.append(complex_seg)

            # Assemble original image using complex data
            # For complex_amplitude_real, the complex segments have the same number of channels as original segments
            original_image = _assemble_image(
                complex_original_segments,
                actual_channels,  # Use actual channels from segments
                nsamples_per_rows,
                nsamples_per_cols,
                patch_size,
                collected_original_indices,
            )
        else:
            # For real_real, use as-is with actual channels from data
            original_image = _assemble_image(
                original_segments,
                actual_channels,
                nsamples_per_rows,
                nsamples_per_cols,
                patch_size,
                collected_original_indices,
            )
    else:
        # For complex modes, use original assembly with actual channels
        original_image = _assemble_image(
            original_segments,
            actual_channels,
            nsamples_per_rows,
            nsamples_per_cols,
            patch_size,
            collected_original_indices,
        )

    # collect reconstructed patches using mini-batch processing
    reconstructed_segments: List[np.ndarray] = []
    collected_reconstructed_indices: List[int] = []

    with torch.no_grad():
        for data in full_loader:
            inputs = data[0] if isinstance(data, (tuple, list)) else data
            outputs = model(inputs.to(device))
            reconstructed_segments.extend([seg for seg in outputs.cpu().numpy()])

            # Extract indices from the same batch
            if len(data) >= 2:

                indices = data[1] if len(data) == 2 else data[2]
                collected_reconstructed_indices.extend(indices.cpu().detach().numpy())

    # Determine output channels for reconstructed image
    if reconstructed_segments:
        output_channels = reconstructed_segments[0].shape[0]
    else:
        output_channels = num_channels

        # Verify that segments and indices have the same length
    if collected_original_indices is not None and len(original_segments) != len(collected_original_indices):
        logger.warning(
            f"Mismatch between segments ({len(original_segments)}) and indices ({len(collected_original_indices)}). "
            f"Using sequential placement."
        )
        collected_original_indices = None  # Fall back to sequential placement

    # Verify that reconstructed segments and indices have the same length  
    if len(collected_reconstructed_indices) == 0:
        # No indices provided by DataLoader, use sequential placement
        collected_reconstructed_indices = None
    elif len(reconstructed_segments) != len(collected_reconstructed_indices):
        logger.warning(
            f"Mismatch between reconstructed segments ({len(reconstructed_segments)}) and indices ({len(collected_reconstructed_indices)}). "
            f"Using sequential placement."
        )
        collected_reconstructed_indices = None  # Fall back to sequential placement

    reconstructed_image = _assemble_image(
        reconstructed_segments,
        output_channels,
        nsamples_per_rows,
        nsamples_per_cols,
        patch_size,
        collected_reconstructed_indices,
    )

    # Convert dual real back to complex if needed
    if layer_mode == "real" and real_pipeline_type == "complex_dual_real":
        # Split dual real back to complex (6 channels -> 3 complex channels)
        if output_channels % 2 != 0:
            raise ValueError(
                f"Dual real output must have even channels, got {output_channels}"
            )
        C = output_channels // 2
        real_part = reconstructed_image[:C]
        imag_part = reconstructed_image[C:]
        reconstructed_image = real_part + 1j * imag_part
    elif layer_mode == "real" and real_pipeline_type == "complex_amplitude_real":
        # For amplitude pipeline, create approximate complex (zero phase)
        reconstructed_image = reconstructed_image + 1j * np.zeros_like(
            reconstructed_image
        )
    # For real_real and complex modes, no conversion needed

    return original_image, reconstructed_image


def create_dataset_split_mask(
    cfg: Dict[str, Any],
    full_loader: torch.utils.data.DataLoader,
    train_indices: List[int],
    valid_indices: List[int],
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    test_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Create dataset split mask similar to plot_dataset_split_mask but return the mask array.
    """
    # Extract necessary parameters from config

    correct_indice_tensors = []
    for data in full_loader:
        correct_indice_tensors.extend(
            data[1].cpu().detach().numpy()
            if len(data) == 2
            else data[2].cpu().detach().numpy()
        )

    patch_size = cfg["data"]["patch_size"]

    nb_rows = nsamples_per_rows * patch_size
    nb_cols = nsamples_per_cols * patch_size

    # Create mask initialized to -1 (no data)
    mask = np.full((nb_rows, nb_cols), -1, dtype=np.int32)

    # Create sets for quick lookup
    sets_indices = [set(train_indices), set(valid_indices)]
    if test_indices is not None:
        sets_indices.append(set(test_indices))

    # Place each patch according to its real_index
    for segment_index, real_index in enumerate(correct_indice_tensors):
        # Convert real_index to grid position
        row = real_index // nsamples_per_cols
        col = real_index % nsamples_per_cols

        # Check bounds
        if row >= nsamples_per_rows or col >= nsamples_per_cols:
            continue  # Skip out-of-bounds indices

        # Calculate pixel coordinates
        h_start = row * patch_size
        w_start = col * patch_size

        # Determine which set this patch belongs to and assign color
        if real_index in sets_indices[0]:  # train
            mask[h_start : h_start + patch_size, w_start : w_start + patch_size] = 0
        elif real_index in sets_indices[1]:  # validation
            mask[h_start : h_start + patch_size, w_start : w_start + patch_size] = 1
        elif len(sets_indices) > 2 and real_index in sets_indices[2]:  # test
            mask[h_start : h_start + patch_size, w_start : w_start + patch_size] = 2

    return mask


def reconstruct_full_segmentation(
    model: torch.nn.Module,
    full_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    real_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct full image segmentation by assembling patch predictions.

    Args:
        model: The trained model for segmentation
        full_loader: DataLoader containing image patches
        config: Configuration dictionary
        device: Device to run inference on
        nsamples_per_rows: Number of patch rows
        nsamples_per_cols: Number of patch columns
        real_indices: Optional list of real indices. If None, will be extracted from the dataloader.

    Returns:
        tuple: (original_image, ground_truth, predicted_segmentation)
    """

    # Collect patches, predictions, ground truth, and indices together
    original_patches: List[np.ndarray] = []
    predicted_patches: List[np.ndarray] = []
    ground_truth_patches: List[np.ndarray] = []
    collected_indices: List[int] = []

    with torch.no_grad():
        for batch in full_loader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            inputs = inputs.to(device)

            # Get predictions
            outputs_non_projected, outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)  # Get class predictions

            # Store patches
            original_patches.extend([patch.cpu().numpy() for patch in inputs])
            predicted_patches.extend([pred.cpu().numpy() for pred in predictions])

            # Extract ground truth labels if available
            if len(batch) >= 2:
                if len(batch) == 2:
                    # batch = [inputs, labels]
                    labels = batch[1]
                    indices = None
                else:
                    # batch = [inputs, labels, indices] or [inputs, indices, labels]
                    if batch[1].dim() > 1:  # labels are usually 2D+ (segmentation maps)
                        labels = batch[1]
                        indices = batch[2] if len(batch) > 2 else None
                    else:  # indices are usually 1D
                        indices = batch[1]
                        labels = batch[2] if len(batch) > 2 else None

                if labels is not None:
                    ground_truth_patches.extend(
                        [label.cpu().numpy() for label in labels]
                    )

                if indices is not None:
                    collected_indices.extend(indices.cpu().detach().numpy())

    # Use collected indices if real_indices not provided
    if real_indices is None:
        real_indices = collected_indices

    # Verify that patches and indices have the same length
    if len(original_patches) != len(real_indices):
        logger.warning(
            f"Mismatch between patches ({len(original_patches)}) and indices ({len(real_indices)}). "
            f"Using sequential placement."
        )
        real_indices = None  # Fall back to sequential placement

    # Get image dimensions from config
    seg_size = config["data"]["patch_size"]

    # Assemble original image
    original_image = _assemble_image(
        original_patches,
        config["data"]["inferred_input_channels"],
        nsamples_per_rows,
        nsamples_per_cols,
        seg_size,
        real_indices,
    )

    # Assemble predicted segmentation
    predicted_segmentation = _assemble_segmentation_image(
        predicted_patches, nsamples_per_rows, nsamples_per_cols, seg_size, real_indices
    )

    # Assemble ground truth segmentation if available
    if ground_truth_patches:
        ground_truth_segmentation = _assemble_segmentation_image(
            ground_truth_patches,
            nsamples_per_rows,
            nsamples_per_cols,
            seg_size,
            real_indices,
        )
    else:
        # Create a dummy ground truth if not available
        logger.warning(
            "No ground truth labels found in dataloader. Creating dummy ground truth."
        )
        ground_truth_segmentation = np.zeros_like(predicted_segmentation)

    return original_image, ground_truth_segmentation, predicted_segmentation


def _assemble_image(
    segments: List[np.ndarray],
    num_channels: int,
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    patch_size: int,
    real_indices: Optional[List[int]] = None,
) -> np.ndarray:

    # Calculate actual image dimensions in pixels
    nb_rows = nsamples_per_rows * patch_size
    nb_cols = nsamples_per_cols * patch_size

    image = np.zeros((num_channels, nb_rows, nb_cols), dtype=segments[0].dtype)

    if real_indices is None:
        # Fall back to sequential placement if no real_indices provided
        idx = 0
        for h in range(0, nb_rows, patch_size):
            for w in range(0, nb_cols, patch_size):
                if (
                    h + patch_size <= nb_rows
                    and w + patch_size <= nb_cols
                    and idx < len(segments)
                ):
                    image[:, h : h + patch_size, w : w + patch_size] = segments[idx]
                    idx += 1
    else:
        # Use real_indices to map segments to their correct positions
        # real_indices[i] tells us the grid position where segments[i] should be placed

        # Place each segment into the correct position
        for segment_index, real_index in enumerate(real_indices):
            if segment_index >= len(segments):
                break

            # Calculate row and col from real_index (sequential patch numbering)
            row = real_index // nsamples_per_cols
            col = real_index % nsamples_per_cols

            # Check bounds
            if row >= nsamples_per_rows or col >= nsamples_per_cols:
                raise ValueError(
                    f"Real index {real_index} maps to position ({row}, {col}) "
                    f"which is out of bounds for grid ({nsamples_per_rows}, {nsamples_per_cols})"
                )

            # Calculate pixel coordinates
            h_start = row * patch_size
            w_start = col * patch_size

            # Ensure we don't exceed image boundaries
            if h_start + patch_size <= nb_rows and w_start + patch_size <= nb_cols:
                image[
                    :, h_start : h_start + patch_size, w_start : w_start + patch_size
                ] = segments[segment_index]

    return image


def _assemble_segmentation_image(
    patches: List[np.ndarray],
    nsamples_per_rows: int,
    nsamples_per_cols: int,
    patch_size: int,
    real_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Assemble segmentation patches into full image."""
    # Calculate actual image dimensions
    nb_rows = nsamples_per_rows * patch_size
    nb_cols = nsamples_per_cols * patch_size

    image = np.zeros((nb_rows, nb_cols), dtype=patches[0].dtype)

    if real_indices is None:
        # Fall back to sequential placement if no real_indices provided
        idx = 0
        for h in range(0, nb_rows, patch_size):
            for w in range(0, nb_cols, patch_size):
                if (
                    h + patch_size <= nb_rows
                    and w + patch_size <= nb_cols
                    and idx < len(patches)
                ):
                    image[h : h + patch_size, w : w + patch_size] = patches[idx]
                    idx += 1
    else:
        # Use real_indices to map patches to their correct positions
        for patch_index, real_index in enumerate(real_indices):
            if patch_index >= len(patches):
                break

            # Calculate row and col from real_index (sequential patch numbering)
            row = real_index // nsamples_per_cols
            col = real_index % nsamples_per_cols

            # Check bounds
            if row >= nsamples_per_rows or col >= nsamples_per_cols:
                continue  # Skip out-of-bounds indices

            # Calculate pixel coordinates
            h_start = row * patch_size
            w_start = col * patch_size

            # Ensure we don't exceed image boundaries
            if h_start + patch_size <= nb_rows and w_start + patch_size <= nb_cols:
                image[
                    h_start : h_start + patch_size, w_start : w_start + patch_size
                ] = patches[patch_index]

    return image


def evaluate_classification(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Evaluate classification performance.
    Returns metrics including accuracy, confusion matrix, etc.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            logger.warning("Model has no parameters; defaulting to CPU")
            device = torch.device("cpu")

    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())  # Calculate metrics
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import numpy as np

    accuracy = accuracy_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, output_dict=True)

    # Safely extract macro averages
    macro_avg = report.get("macro avg", {}) if isinstance(report, dict) else {}

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "macro_precision": (
            macro_avg.get("precision", 0.0) if isinstance(macro_avg, dict) else 0.0
        ),
        "macro_recall": (
            macro_avg.get("recall", 0.0) if isinstance(macro_avg, dict) else 0.0
        ),
        "macro_f1": (
            macro_avg.get("f1-score", 0.0) if isinstance(macro_avg, dict) else 0.0
        ),
    }

    if wandb.run:
        wandb.log({"classification/accuracy": accuracy})

    return metrics


def evaluate_segmentation(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Evaluate segmentation performance.
    Returns metrics including IoU, pixel accuracy, etc.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            logger.warning("Model has no parameters; defaulting to CPU")
            device = torch.device("cpu")

    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []
    shift_consistency_total = 0.0

    # Store samples for visualization (limit to avoid memory issues)
    sample_images = []
    sample_ground_truth = []
    sample_predictions = []
    max_visualization_samples = 5

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, outputs_projected = model(inputs)
                predictions = torch.argmax(outputs_projected, dim=1)

                # Compute circular shift consistency for this batch
                batch_consistency = circular_shift_consistency(model, inputs, "segmentation", device)
                shift_consistency_total += batch_consistency

                # Store flattened data for metrics
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

                # Store sample data for visualization (first few batches)
                if batch_idx < max_visualization_samples:
                    # Take first sample from batch for visualization
                    sample_images.append(inputs[0].cpu().numpy())
                    sample_ground_truth.append(labels[0].cpu().numpy())
                    sample_predictions.append(predictions[0].cpu().numpy())

    # Calculate enhanced metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )
    import numpy as np

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Handle empty data case
    if len(all_labels) == 0 or len(all_predictions) == 0:
        return {
            "pixel_accuracy": 0.0,
            "mean_iou": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "weighted_accuracy": 0.0,
            "sample_images": [],
            "sample_ground_truth": [],
            "sample_predictions": []
        }
    
    # Ensure labels are integers for classification metrics
    if all_labels.dtype in [np.complex64, np.complex128]:
        # For complex labels, use magnitude and convert to int
        all_labels = np.abs(all_labels).astype(int)
    elif all_labels.dtype in [np.float32, np.float64]:
        # For float labels, convert to int
        all_labels = all_labels.astype(int)

    # Basic pixel accuracy
    pixel_accuracy = accuracy_score(all_labels, all_predictions)

    # Get unique classes from both predictions and labels
    all_classes = np.unique(np.concatenate([all_labels, all_predictions]))
    num_classes = len(all_classes)

    # Per-class precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, labels=all_classes, average=None, zero_division=0
    )

    # Class frequency for weighting
    class_frequencies = np.bincount(all_labels, minlength=num_classes)
    class_frequencies = class_frequencies / class_frequencies.sum()

    # Class-weighted metrics (weighted by inverse frequency)
    sample_weights = 1.0 / (
        class_frequencies[all_labels] + 1e-8
    )  # avoid division by zero
    weighted_accuracy = accuracy_score(
        all_labels, all_predictions, sample_weight=sample_weights
    )

    # Macro and weighted averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)

    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)

    # IoU calculation (per class and variants)
    iou_per_class = []
    dice_per_class = []

    for class_id in all_classes:
        pred_mask = all_predictions == class_id
        true_mask = all_labels == class_id
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()

        # IoU
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        iou_per_class.append(iou)

        # Dice coefficient
        true_positive = intersection
        false_positive = pred_mask.sum() - intersection
        false_negative = true_mask.sum() - intersection

        if (true_positive + false_positive + false_negative) == 0:
            dice = 1.0
        else:
            dice = (2 * true_positive) / (
                2 * true_positive + false_positive + false_negative
            )
        dice_per_class.append(dice)

    mean_iou = np.mean(iou_per_class)
    mean_dice = np.mean(dice_per_class)

    # Frequency-weighted IoU
    frequency_weighted_iou = np.average(
        iou_per_class, weights=class_frequencies[: len(iou_per_class)]
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=all_classes)

    metrics = {
        # Basic metrics
        "pixel_accuracy": pixel_accuracy,
        "weighted_accuracy": weighted_accuracy,
        # IoU metrics
        "mean_iou": mean_iou,
        "frequency_weighted_iou": frequency_weighted_iou,
        "iou_per_class": iou_per_class,
        # Dice metrics
        "mean_dice": mean_dice,
        "dice_per_class": dice_per_class,
        # Per-class metrics
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        # Macro averages
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        # Weighted averages
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        # Additional data
        "confusion_matrix": conf_matrix.tolist(),
        "class_frequencies": class_frequencies.tolist(),
        "num_classes": num_classes,
        # Circular shift consistency
        "circular_shift_consistency": shift_consistency_total / len(test_loader),
        # Visualization data
        "sample_images": sample_images,
        "sample_ground_truth": sample_ground_truth,
        "sample_predictions": sample_predictions,
    }

    logger.info(f"Segmentation Circular Shift Consistency: {metrics['circular_shift_consistency']:.4f}")

    if wandb.run:
        wandb_log_dict = {
            # Basic accuracy metrics
            "segmentation/pixel_accuracy": pixel_accuracy,
            "segmentation/weighted_accuracy": weighted_accuracy,
            # IoU metrics
            "segmentation/mean_iou": mean_iou,
            "segmentation/frequency_weighted_iou": frequency_weighted_iou,
            # Dice metrics
            "segmentation/mean_dice": mean_dice,
            # Macro averages
            "segmentation/precision_macro": precision_macro,
            "segmentation/recall_macro": recall_macro,
            "segmentation/f1_macro": f1_macro,
            # Weighted averages
            "segmentation/precision_weighted": precision_weighted,
            "segmentation/recall_weighted": recall_weighted,
            "segmentation/f1_weighted": f1_weighted,
        }

        # Add circular shift consistency
        wandb_log_dict["segmentation/circular_shift_consistency"] = metrics["circular_shift_consistency"]

        # Add per-class metrics
        for i, (iou, dice, prec, rec, f1_score) in enumerate(
            zip(iou_per_class, dice_per_class, precision, recall, f1)
        ):
            wandb_log_dict.update(
                {
                    f"segmentation/class_{i}/iou": iou,
                    f"segmentation/class_{i}/dice": dice,
                    f"segmentation/class_{i}/precision": prec,
                    f"segmentation/class_{i}/recall": rec,
                    f"segmentation/class_{i}/f1": f1_score,
                }
            )

        wandb.log(wandb_log_dict)

    return metrics


# def evaluate_generation(
#     test_loader: torch.utils.data.DataLoader,
#     model: torch.nn.Module,
#     device: torch.device = None,
# ) -> Dict[str, Any]:
#     """
#     Evaluate generation quality.
#     Returns metrics for generated samples.
#     """
#     if device is None:
#         try:
#             device = next(model.parameters()).device
#         except StopIteration:
#             logger.warning("Model has no parameters; defaulting to CPU")
#             device = torch.device("cpu")

#     model.eval()
#     model.to(device)

#     generated_samples = []
#     original_samples = []

#     with torch.no_grad():
#         for batch in test_loader:
#             inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
#             inputs = inputs.to(device)

#             # Generate samples
#             outputs = model(inputs)

#             generated_samples.extend(outputs.cpu().numpy())
#             original_samples.extend(inputs.cpu().numpy())

#     # Calculate generation metrics
#     generated_samples = np.array(generated_samples)
#     original_samples = np.array(original_samples)

#     # Simple MSE between generated and original
#     mse = np.mean((generated_samples - original_samples) ** 2)

#     metrics = {
#         "generation_mse": mse,
#         "num_generated_samples": len(generated_samples),
#         "sample_shape": (
#             generated_samples.shape[1:] if len(generated_samples) > 0 else None
#         ),
#     }

#     if wandb.run:
#         wandb.log({"generation/mse": mse})

#     return metrics


def evaluate_complex_ssim_methods(original, reconstructed, window_size=11):
    """
    Compare different complex SSIM methods and return detailed metrics.

    Args:
        original, reconstructed: Complex-valued tensors
        window_size: SSIM window size

    Returns:
        Dictionary with different SSIM scores and analysis
    """
    results = {}

    # Original simple approach (for comparison)
    mag_ssim = ssim(torch.abs(original), torch.abs(reconstructed), window_size)
    phase_ssim = ssim(torch.angle(original), torch.angle(reconstructed), window_size)
    results["simple_average"] = (mag_ssim.item() + phase_ssim.item()) / 2
    results["magnitude_only"] = mag_ssim.item()
    results["phase_only"] = phase_ssim.item()

    # Improved methods
    results["complex_ssim"] = complex_ssim(original, reconstructed, window_size).item()
    results["complex_correlation"] = complex_ssim_correlation(
        original, reconstructed, window_size
    ).item()

    # Advanced method (more computationally expensive)
    try:
        results["complex_advanced"] = complex_ssim_advanced(
            original, reconstructed, window_size
        ).item()
    except Exception as e:
        logger.warning(f"Advanced complex SSIM failed: {e}")
        results["complex_advanced"] = None

    # Analysis metrics
    mag_mean = torch.mean(torch.abs(original)).item()
    mag_std = torch.std(torch.abs(original)).item()
    phase_consistency = torch.mean(
        torch.cos(torch.angle(original) - torch.angle(reconstructed))
    ).item()

    results["analysis"] = {
        "magnitude_mean": mag_mean,
        "magnitude_std": mag_std,
        "phase_consistency": phase_consistency,
        "dynamic_range": torch.max(torch.abs(original)).item()
        / (torch.min(torch.abs(original)).item() + 1e-8),
    }

    return results


def adaptive_complex_ssim(original, reconstructed, window_size=11):
    """
    Adaptive complex SSIM that chooses the best method based on image characteristics.

    Args:
        original, reconstructed: Complex-valued tensors
        window_size: SSIM window size

    Returns:
        Adaptive SSIM score and method used
    """
    # Analyze image characteristics
    mag_original = torch.abs(original)
    mag_reconstructed = torch.abs(reconstructed)

    # Check image size and adjust window size if necessary
    B, C, H, W = original.shape
    effective_window_size = min(window_size, min(H, W))

    # Calculate SNR-like metric
    signal_power = torch.mean(mag_original**2)
    noise_power = torch.mean((mag_original - mag_reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

    # Calculate phase importance (higher when magnitudes are significant)
    phase_importance = torch.mean(
        (mag_original > 0.1 * torch.max(mag_original)).float()
    )

    # Adaptive method selection
    if snr > 20 and phase_importance > 0.5 and H >= window_size and W >= window_size:
        # High SNR and significant phase content and large enough image - use advanced method
        method = "advanced"
        score = complex_ssim_advanced(original, reconstructed, effective_window_size)
    elif phase_importance > 0.3:
        # Moderate phase content - use correlation method
        method = "correlation"
        score = complex_ssim_correlation(original, reconstructed, effective_window_size)
    else:
        # Low phase content or small image - use magnitude-weighted method
        method = "magnitude_weighted"
        score = complex_ssim(
            original,
            reconstructed,
            effective_window_size,
            magnitude_weight=0.85,
            phase_weight=0.15,
        )

    return score.item(), method


def circular_shift_consistency(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    task: str,
    device: torch.device = None,
    fixed_shift: tuple = None,
) -> float:
    """
    Measure circular shift consistency of model predictions.
    
    Tests translation equivariance by applying circular shifts to input,
    getting model predictions, then unshifting and comparing with original predictions.
    
    Args:
        model: The model to test
        inputs: Input tensor batch
        task: Task type - "reconstruction" (MSE) or "segmentation" (top-1 agreement)
        device: Device to run computation on
        fixed_shift: Optional fixed shift (off0, off1). If None, uses random shifts.
        
    Returns:
        Consistency score (lower is better for reconstruction, agreement ratio for segmentation)
    """
    if device is None:
        device = inputs.device
    
    model.eval()
    model = model.to(device)
    inputs = inputs.to(device)
    
    # Use fixed shift for debugging, or random shifts
    if fixed_shift is not None:
        off0, off1 = fixed_shift
    else:
        # Random shifts between 1-8 pixels in both dimensions
        off0 = np.random.randint(1, 9)
        off1 = np.random.randint(1, 9)
    
    if task == "reconstruction":
        with torch.no_grad():
            # Forward pass on original input
            output1 = model(inputs)
            
            # Shift input
            shifted_input = torch.roll(inputs, shifts=(off0, off1), dims=(-1, -2))
            output2 = model(shifted_input)
            
            # Shift output1 by same amount
            shifted_output1 = torch.roll(output1, shifts=(off0, off1), dims=(-1, -2))
            
            # Compare shifted_output1 with output2
            consistency = torch.norm(shifted_output1 - output2).item()
        
    elif task == "segmentation":
        # Get original predictions
        with torch.no_grad():
            original_pred_non_projected, original_pred = model(inputs)
        
        # Apply circular shift to inputs
        inputs_shifted = torch.roll(inputs, shifts=(off0, off1), dims=(-1, -2))

        # Get predictions on shifted inputs
        with torch.no_grad():
            shifted_pred_non_projected, shifted_pred = model(inputs_shifted)
        
        # Unshift the predictions to align with original
        unshifted_pred = torch.roll(shifted_pred, shifts=(-off0, -off1), dims=(-1, -2))

        # For segmentation: compute agreement between top-1 predictions
        original_top1 = torch.argmax(original_pred, dim=1)
        unshifted_top1 = torch.argmax(unshifted_pred, dim=1)
        
        #Compute pixel-wise agreement
        agreement = (original_top1 == unshifted_top1).float()
        consistency = agreement.mean().item()
        
    else:
        raise ValueError(f"Unsupported task: {task}. Must be 'reconstruction' or 'segmentation'")
    
    return consistency
