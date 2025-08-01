# Standard library imports
import pathlib
import random
from typing import List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, Subset
from torchcvnn.datasets import ALOSDataset, Bretigny, PolSFDataset

# Local imports
from cvnn.transform_registry import build_transform_pipeline
from cvnn.utils import setup_logging

# module-level logger
logger = setup_logging(__name__)


class Sethi(Dataset):
    def __init__(
        self,
        root: str,
        transform=None,
        patch_size: tuple = (128, 128),
        patch_stride: tuple = None,
        crop_coordinates: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ):
        self.root = root
        self.transform = transform
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride else patch_size

        # alos2_url = "https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip"
        # labels_url = "https://raw.githubusercontent.com/liuxuvip/PolSF/master/SF-ALOS2/SF-ALOS2-label2d.png"
        root = pathlib.Path(root)

        self.HH = np.load(root / "HH.npy")
        self.HV = np.load(root / "HV.npy")
        self.VH = np.load(root / "VH.npy")
        self.VV = np.load(root / "VV.npy")

        if crop_coordinates is not None:
            self.crop_coordinates = crop_coordinates
            self.HH = self.HH[self.crop_coordinates[0][0]:self.crop_coordinates[1][0],
                              self.crop_coordinates[0][1]:self.crop_coordinates[1][1]]
            self.HV = self.HV[self.crop_coordinates[0][0]:self.crop_coordinates[1][0],
                              self.crop_coordinates[0][1]:self.crop_coordinates[1][1]]
            self.VH = self.VH[self.crop_coordinates[0][0]:self.crop_coordinates[1][0],
                              self.crop_coordinates[0][1]:self.crop_coordinates[1][1]]
            self.VV = self.VV[self.crop_coordinates[0][0]:self.crop_coordinates[1][0],
                              self.crop_coordinates[0][1]:self.crop_coordinates[1][1]]   
        else:
            self.crop_coordinates = (
                (0, 0),  # start_row, start_col
                (self.HH.shape[0], self.HH.shape[1]),  # end_row, end_col
            )

        # Precompute the dimension of the grid of patches
        nrows = self.crop_coordinates[1][0] - self.crop_coordinates[0][0]
        ncols = self.crop_coordinates[1][1] - self.crop_coordinates[0][1]

        nrows_patch, ncols_patch = self.patch_size
        row_stride, col_stride = self.patch_stride

        self.nsamples_per_rows = (nrows - nrows_patch) // row_stride + 1
        self.nsamples_per_cols = (ncols - ncols_patch) // col_stride + 1

    def __len__(self) -> int:
        """
        Returns the total number of patches in the while image.

        Returns:
            the total number of patches in the dataset
        """
        return self.nsamples_per_rows * self.nsamples_per_cols

    def __getitem__(self, idx) -> Tuple:
        """
        Returns the indexes patch.

        Arguments:
            idx (int): Index

        Returns:
            tuple: (patch, labels) where patch contains the 4 complex valued polarization HH, HV, VH, VV and labels contains the aligned semantic labels
        """
        row_stride, col_stride = self.patch_stride
        start_row = (idx // self.nsamples_per_cols) * row_stride
        start_col = (idx % self.nsamples_per_cols) * col_stride
        num_rows, num_cols = self.patch_size
        patches = [
            patch[
                start_row : (start_row + num_rows), start_col : (start_col + num_cols)
            ]
            for patch in [self.HH, self.HV, self.VH, self.VV]
        ]
        patches = np.stack(patches)

        if self.transform is not None:
            patches = self.transform(patches)

        return patches


class GenericDatasetWrapper(Dataset):
    def __init__(self, dataset):
        """
        A generic dataset wrapper that works with any dataset class.

        Args:
            dataset: An instance of a dataset class (e.g., CIFAR10, MNIST, etc.).
        """
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Fetch an item from the dataset.

        Args:
            index: Index of the item to fetch.

        Returns:
            A tuple containing (data, target, index).
        """
        data = self.dataset[index]
        if isinstance(data, tuple):
            return data[0], data[1], index
        else:
            return data, index

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.dataset)


def _parse_dataset_config(cfg: dict) -> dict:
    """
    Parse and extract common dataset configuration parameters.

    Args:
        cfg: Configuration dictionary

    Returns:
        Dictionary containing parsed configuration parameters
    """
    dataset_name = cfg["data"]["dataset"]["name"]

    config = {
        "dataset_name": dataset_name,
        "trainpath": cfg["data"]["dataset"]["trainpath"],
        "patch_size": cfg["data"]["patch_size"],
        "patch_stride": cfg["data"].get("patch_stride", cfg["data"]["patch_size"]),
    }

    # Add crop coordinates if available
    if "crop" in cfg["data"]:
        config["crop_coordinates"] = (
            (cfg["data"]["crop"]["start_row"], cfg["data"]["crop"]["start_col"]),
            (cfg["data"]["crop"]["end_row"], cfg["data"]["crop"]["end_col"]),
        )

    return config


def _create_dataset(cfg: dict, transform=None, dataset_config: dict = None):
    """
    Create a dataset instance based on configuration.

    Args:
        cfg: Full configuration dictionary
        transform: Transform to apply to the dataset
        dataset_config: Parsed dataset configuration (optional, will parse if not provided)

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset name is unknown
        FileNotFoundError: If required paths don't exist
    """
    if dataset_config is None:
        dataset_config = _parse_dataset_config(cfg)

    dataset_name = dataset_config["dataset_name"]

    if dataset_name == "Bretigny":
        # Bretigny uses fold-based splits, so we need to specify which fold
        # This function creates a generic dataset; specific folds handled in calling code
        raise NotImplementedError(
            "Bretigny requires fold specification - use _create_bretigny_dataset"
        )
    elif dataset_name == "Sethi":
        return Sethi(
            root=dataset_config["trainpath"],
            transform=transform,
            patch_size=(dataset_config["patch_size"], dataset_config["patch_size"]),
            patch_stride=(
                dataset_config["patch_stride"],
                dataset_config["patch_stride"],
            ),
            crop_coordinates=dataset_config.get("crop_coordinates"),
        )        

    elif dataset_name == "ALOSDataset":
        base_path = pathlib.Path(dataset_config["trainpath"])
        vol_folder = "VOL-ALOS2044980750-150324-HBQR1.1__A"
        trainpath = _find_volpath(base_path, vol_folder)

        return ALOSDataset(
            volpath=trainpath,
            transform=transform,
            crop_coordinates=dataset_config.get("crop_coordinates"),
            patch_size=(dataset_config["patch_size"], dataset_config["patch_size"]),
            patch_stride=(
                dataset_config["patch_stride"],
                dataset_config["patch_stride"],
            ),
        )

    elif dataset_name == "PolSFDataset":
        return PolSFDataset(
            root=dataset_config["trainpath"],
            transform=transform,
            patch_size=(dataset_config["patch_size"], dataset_config["patch_size"]),
            patch_stride=(
                dataset_config["patch_stride"],
                dataset_config["patch_stride"],
            ),
        )

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def _create_bretigny_dataset(
    cfg: dict, fold: str, transform=None, dataset_config: dict = None
):
    """
    Create a Bretigny dataset instance for a specific fold.

    Args:
        cfg: Full configuration dictionary
        fold: Fold name ('train', 'valid', 'test')
        transform: Transform to apply to the dataset
        dataset_config: Parsed dataset configuration (optional)

    Returns:
        Bretigny dataset instance for the specified fold
    """
    if dataset_config is None:
        dataset_config = _parse_dataset_config(cfg)

    return Bretigny(
        root=dataset_config["trainpath"],
        fold=fold,
        transform=transform,
        patch_size=(dataset_config["patch_size"], dataset_config["patch_size"]),
        patch_stride=(dataset_config["patch_stride"], dataset_config["patch_stride"]),
        keep_labels= True if cfg["task"] == "segmentation" else False,  # Keep labels only for segmentation tasks
    )


# Dataset Type Registry
DATASET_TYPE_REGISTRY = {
    "ALOSDataset": {
        "type": "polsar",
        "supports_full_image_reconstruction": True,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": False,
        "ignore_index": None,
    },
    "PolSFDataset": {
        "type": "polsar",
        "supports_full_image_reconstruction": True,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": True,
        "ignore_index": 0,
    },
    "Sethi": {
        "type": "polsar",
        "supports_full_image_reconstruction": True,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": False,
        "ignore_index": None,
    },
    "Bretigny": {
        "type": "polsar",
        "supports_full_image_reconstruction": True,
        "valid_layer_modes": ["complex", "split", "real"],
        "valid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "invalid_real_pipelines": ["real_real"],
        "default_real_pipeline": "complex_dual_real",
        "has_labels": True,
        "ignore_index": 0,
    },
    # Add more datasets as needed
    "MNIST": {
        "type": "grayscale",
        "supports_full_image_reconstruction": False,
        "valid_layer_modes": ["real"],
        "valid_real_pipelines": ["real_real"],
        "invalid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "default_real_pipeline": "real_real",
        "has_labels": True,
        "ignore_index": None,
    },
    "CIFAR10": {
        "type": "rgb",
        "supports_full_image_reconstruction": False,
        "valid_layer_modes": ["real"],
        "valid_real_pipelines": ["real_real"],
        "invalid_real_pipelines": ["complex_amplitude_real", "complex_dual_real"],
        "default_real_pipeline": "real_real",
        "has_labels": True,
        "ignore_index": None,
    },
}


def get_dataset_info(dataset_name: str) -> dict:
    """Get dataset type information from registry.

    Args:
        dataset_name: Name of the dataset class

    Returns:
        Dictionary with dataset type information

    Raises:
        ValueError: If dataset is not registered
    """
    if dataset_name not in DATASET_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Registered datasets: {list(DATASET_TYPE_REGISTRY.keys())}"
        )
    return DATASET_TYPE_REGISTRY[dataset_name]


def get_dataset_type_from_config(cfg: dict) -> str:
    """Get dataset type from configuration using DATASET_TYPE_REGISTRY.

    Args:
        cfg: Configuration dictionary containing dataset information

    Returns:
        Dataset type string (e.g., 'polsar', 'grayscale', 'rgb')

    Raises:
        ValueError: If dataset is not found in registry
    """
    dataset_name = cfg["data"]["dataset"]["name"]
    return get_dataset_info(dataset_name)["type"]


def validate_and_correct_config(cfg: dict) -> dict:
    """Validate and auto-correct configuration based on dataset type.

    Args:
        cfg: Configuration dictionary

    Returns:
        Corrected configuration dictionary
    """
    # Check if required keys exist
    if (
        "data" not in cfg
        or "dataset" not in cfg["data"]
        or "name" not in cfg["data"]["dataset"]
    ):
        return cfg

    if "model" not in cfg or "layer_mode" not in cfg["model"]:
        return cfg

    dataset_name = cfg["data"]["dataset"]["name"]
    dataset_info = get_dataset_info(dataset_name)

    layer_mode = cfg["model"]["layer_mode"]
    real_pipeline_type = cfg["data"].get("real_pipeline_type")

    # Validate layer_mode
    if layer_mode not in dataset_info["valid_layer_modes"]:
        valid_modes = dataset_info["valid_layer_modes"]
        default_mode = valid_modes[0]
        logger.warning(
            f"Invalid layer_mode '{layer_mode}' for {dataset_info['type']} dataset '{dataset_name}'. "
            f"Valid modes: {valid_modes}. Using '{default_mode}' instead."
        )
        cfg["model"]["layer_mode"] = default_mode
        layer_mode = default_mode

    # Handle real pipeline type validation
    if layer_mode == "real":
        if real_pipeline_type is None:
            # Use default for this dataset type
            default_pipeline = dataset_info["default_real_pipeline"]
            logger.warning(
                f"No real_pipeline_type specified for real layer_mode with {dataset_info['type']} dataset. "
                f"Using default: '{default_pipeline}'"
            )
            cfg["data"]["real_pipeline_type"] = default_pipeline
            real_pipeline_type = default_pipeline
        elif real_pipeline_type in dataset_info["invalid_real_pipelines"]:
            # Use default and warn
            default_pipeline = dataset_info["default_real_pipeline"]
            logger.warning(
                f"Invalid real_pipeline_type '{real_pipeline_type}' for {dataset_info['type']} dataset '{dataset_name}'. "
                f"Valid pipelines: {dataset_info['valid_real_pipelines']}. "
                f"Using '{default_pipeline}' instead."
            )
            cfg["data"]["real_pipeline_type"] = default_pipeline

    # Add ignore_index for datasets with unlabeled pixels
    dataset_info = get_dataset_info(dataset_name)
    if dataset_info.get("ignore_index") is not None:
        if "training" not in cfg:
            cfg["training"] = {}
        cfg["training"]["ignore_index"] = dataset_info["ignore_index"]
        logger.info(
            f"Auto-configured ignore_index={dataset_info['ignore_index']} for {dataset_name} (unlabeled pixels)"
        )

    return cfg


def infer_channels_from_dataloader(dataloader: DataLoader) -> int:
    """Infer number of channels by sampling from the dataloader.

    Args:
        dataloader: PyTorch DataLoader to sample from

    Returns:
        Number of input channels

    Raises:
        ValueError: If data shape is unexpected
    """
    sample_batch = next(iter(dataloader))

    # Handle different data formats
    if isinstance(sample_batch, dict):
        # Dictionary format (should not occur anymore after simplification)
        if "data" in sample_batch:
            sample_data = sample_batch["data"]
        else:
            sample_data = list(sample_batch.values())[0]
    elif isinstance(sample_batch, (list, tuple)):
        sample_data = sample_batch[0]  # First element is usually the input
    else:
        sample_data = sample_batch

    if len(sample_data.shape) == 4:  # [batch, channels, height, width]
        return sample_data.shape[1]
    elif len(sample_data.shape) == 3:  # [channels, height, width]
        return sample_data.shape[0]
    else:
        raise ValueError(f"Unexpected data shape: {sample_data.shape}")


def _find_volpath(
    base_path: pathlib.Path, vol_folder: str, max_up: int = 5
) -> pathlib.Path:
    """Recursively search parent directories for the given vol_folder under base_path."""
    # pre-define candidate for type-safety
    candidate = base_path / vol_folder
    for i in range(max_up + 1):
        candidate = pathlib.Path("../" * i) / base_path / vol_folder
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find volpath after {max_up} levels: {candidate}"
    )


def get_dataloaders(
    cfg: dict, use_cuda: bool
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    """Create training and validation data loaders based on configuration."""

    # Get appropriate transform using new transform registry
    input_transform = build_transform_pipeline(cfg)

    # Parse dataset configuration once
    dataset_config = _parse_dataset_config(cfg)

    # initialize test split placeholders
    test_indices: Optional[List[int]] = None
    test_dataset = None  # type: Optional[torch.utils.data.Subset]

    if dataset_config["dataset_name"] == "Bretigny":
        train_dataset = _create_bretigny_dataset(
            cfg, "train", input_transform, dataset_config
        )
        valid_dataset = _create_bretigny_dataset(
            cfg, "valid", input_transform, dataset_config
        )
        test_dataset = _create_bretigny_dataset(
            cfg, "test", input_transform, dataset_config
        )

        logger.info(
            f"Loaded {len(train_dataset) + len(valid_dataset) + len(test_dataset)} samples from Bretigny"
        )
    else:
        # Create base dataset using helper function
        base_dataset = _create_dataset(cfg, input_transform, dataset_config)

        logger.info(
            f"Loaded {len(base_dataset)} samples from {dataset_config['dataset_name']}"
        )

        # Use label-based splitting for PolSFDataset (has labels), random for others
        task = cfg.get("task", "")
        if dataset_config["dataset_name"] == "PolSFDataset" and task in [
            "segmentation",
            "classification",
        ]:
            logger.info("Using label-based clustering split for PolSFDataset")
            train_indices, valid_indices, test_indices = get_label_based_split_indices(
                base_dataset, cfg
            )
        else:
            logger.info("Using random split")
            indices = list(range(len(base_dataset)))
            random.shuffle(indices)
            test_ratio = cfg["data"].get("test_ratio", 0.0)
            num_valid = int(cfg["data"]["valid_ratio"] * len(indices))
            num_test = int(test_ratio * len(indices))
            num_train = len(indices) - num_valid - num_test
            train_indices = indices[:num_train]
            valid_indices = indices[num_train : num_train + num_valid]
            test_indices = indices[num_train + num_valid :] if test_ratio > 0 else None

        train_dataset = Subset(base_dataset, train_indices)
        valid_dataset = Subset(base_dataset, valid_indices)
        if test_indices is not None:
            test_dataset = Subset(base_dataset, test_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=use_cuda,
    )
    # optionally include test loader
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=use_cuda,
        )

        # Infer and store the actual input channels in config
        # Use train_loader for channel inference (it has the correct format for model input)
        inferred_channels = infer_channels_from_dataloader(train_loader)
        cfg["data"]["inferred_input_channels"] = inferred_channels
        logger.info(f"Inferred {inferred_channels} input channels from data pipeline")

        # Infer number of classes for segmentation tasks
        task = cfg.get("task", "")
        if task == "segmentation":
            try:
                inferred_classes = infer_classes_from_dataloader(train_loader)
                cfg["model"]["inferred_num_classes"] = inferred_classes
                logger.info(
                    f"Inferred {inferred_classes} classes from segmentation data"
                )
            except Exception as e:
                logger.warning(f"Could not infer number of classes: {e}")

        return train_loader, valid_loader, test_loader

    # Infer and store the actual input channels in config
    inferred_channels = infer_channels_from_dataloader(train_loader)
    cfg["data"]["inferred_input_channels"] = inferred_channels
    logger.info(f"Inferred {inferred_channels} input channels from data pipeline")

    # Infer number of classes for segmentation tasks
    task = cfg.get("task", "")
    if task == "segmentation":
        try:
            inferred_classes = infer_classes_from_dataloader(train_loader)
            cfg["model"]["inferred_num_classes"] = inferred_classes
            logger.info(f"Inferred {inferred_classes} classes from segmentation data")
        except Exception as e:
            logger.warning(f"Could not infer number of classes: {e}")

    return train_loader, valid_loader


def get_full_image_dataloader(cfg: dict, use_cuda: bool) -> Tuple[DataLoader, int, int]:
    """Get a DataLoader for the full image dataset."""

    # Get base transform pipeline only (excluding additional transforms from config)
    input_transform = build_transform_pipeline(cfg)

    # Parse dataset configuration once
    dataset_config = _parse_dataset_config(cfg)
    nsamples_per_cols = 0
    nsamples_per_rows = 0
    dataset_config["patch_stride"] = cfg["data"]["patch_size"] # Use patch size as stride for full image
 
    if dataset_config["dataset_name"] == "Bretigny":
        base_dataset = _create_bretigny_dataset(
            cfg, "all", input_transform, dataset_config
        )
        nsamples_per_cols = base_dataset.nsamples_per_cols
        nsamples_per_rows = base_dataset.nsamples_per_rows

    elif dataset_config["dataset_name"] in ["ALOSDataset", "PolSFDataset"]:
        # Create base dataset using helper function
        # We restrain the full image to a small portion to avoid memory issues
        if "crop_coordinates" not in dataset_config:
            dataset_config["crop_coordinates"] = (
                (0, 0),
                (9000, 5000),
            )
        base_dataset = _create_dataset(cfg, input_transform, dataset_config)
        if dataset_config["dataset_name"] == "ALOSDataset":
            nsamples_per_cols = base_dataset.nsamples_per_cols
            nsamples_per_rows = base_dataset.nsamples_per_rows
        elif dataset_config["dataset_name"] == "PolSFDataset":
            nsamples_per_cols = base_dataset.alos_dataset.nsamples_per_cols
            nsamples_per_rows = base_dataset.alos_dataset.nsamples_per_rows
    elif dataset_config["dataset_name"] == "Sethi":
        if "crop_coordinates" not in dataset_config:
            dataset_config["crop_coordinates"] = (
                (0, 0),
                (9000, 9000),
            )
        # Create Sethi dataset with full image
        base_dataset = _create_dataset(cfg, input_transform, dataset_config)
        nsamples_per_cols = base_dataset.nsamples_per_cols
        nsamples_per_rows = base_dataset.nsamples_per_rows

    logger.info(f"Full image data loader with {len(base_dataset)} segments")
    wrapped_dataset = GenericDatasetWrapper(base_dataset)
    # Build the dataloader
    data_loader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=use_cuda,
    )
    return data_loader, nsamples_per_rows, nsamples_per_cols


def infer_classes_from_dataloader(dataloader: DataLoader) -> int:
    """Infer number of classes by sampling labels from the segmentation dataloader.

    Args:
        dataloader: PyTorch DataLoader to sample from (should contain (input, target) pairs)

    Returns:
        Number of unique classes found in the labels

    Raises:
        ValueError: If data format is unexpected or no labels found
    """
    unique_labels = set()
    max_samples = min(10, len(dataloader))  # Sample up to 10 batches to infer classes

    for i, sample_batch in enumerate(dataloader):
        if i >= max_samples:
            break

        # Handle different data formats for segmentation (input, target) pairs
        if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
            targets = sample_batch[1]  # Second element should be the labels/targets
        elif isinstance(sample_batch, dict) and "targets" in sample_batch:
            targets = sample_batch["targets"]
        elif isinstance(sample_batch, dict) and "labels" in sample_batch:
            targets = sample_batch["labels"]
        else:
            raise ValueError(
                f"Expected (input, target) tuple or dict with 'targets'/'labels', got: {type(sample_batch)}"
            )

        # Convert to numpy for easier processing
        if isinstance(targets, torch.Tensor):
            targets_np = targets.cpu().numpy()
        else:
            targets_np = np.array(targets)

        # Get unique labels from this batch
        batch_unique = np.unique(targets_np)
        unique_labels.update(batch_unique.tolist())

    if not unique_labels:
        raise ValueError("No labels found in the dataset")

    num_classes = len(unique_labels)
    logger.info(
        f"Found {num_classes} unique classes in dataset: {sorted(unique_labels)}"
    )

    return num_classes


def get_dataset_split_indices(
    cfg: dict
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    """
    Get the train/validation/test split indices used by get_dataloaders.

    This function replicates the splitting logic from get_dataloaders but only
    returns the indices, useful for visualization purposes.

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of (train_indices, valid_indices, test_indices).
        test_indices is None if test_ratio is 0.

    Raises:
        ValueError: If dataset configuration is invalid
    """
    # Initialize transform using transform registry
    input_transform = build_transform_pipeline(cfg)

    # Parse dataset configuration once
    dataset_config = _parse_dataset_config(cfg)

    if dataset_config["dataset_name"] == "Bretigny":
        # For Bretigny, we need to get total count across all folds
        train_dataset = _create_bretigny_dataset(
            cfg, "train", input_transform, dataset_config
        )
        valid_dataset = _create_bretigny_dataset(
            cfg, "valid", input_transform, dataset_config
        )
        test_dataset = _create_bretigny_dataset(
            cfg, "test", input_transform, dataset_config
        )

        # For Bretigny, return sequential indices for each fold
        total_train = len(train_dataset)
        total_valid = len(valid_dataset)
        total_test = len(test_dataset)

        train_indices = list(range(total_train))
        valid_indices = list(range(total_train, total_train + total_valid))
        test_indices = list(
            range(total_train + total_valid, total_train + total_valid + total_test)
        )

        return train_indices, valid_indices, test_indices

    else:
        # For other datasets, create base dataset and replicate splitting logic
        try:
            base_dataset = _create_dataset(cfg, input_transform, dataset_config)
        except FileNotFoundError:
            # If dataset path doesn't exist, return empty indices for testing
            logger.warning(f"Dataset path not found. Cannot get split indices.")
            return [], [], None

        # Use same splitting logic as get_dataloaders
        task = cfg.get("task", "")
        if dataset_config["dataset_name"] == "PolSFDataset" and task in [
            "segmentation",
            "classification",
        ]:
            train_indices, valid_indices, test_indices = get_label_based_split_indices(
                base_dataset, cfg
            )
        else:
            # Replicate the exact random splitting logic
            indices = list(range(len(base_dataset)))
            random.shuffle(indices)
            test_ratio = cfg["data"].get("test_ratio", 0.0)
            num_valid = int(cfg["data"]["valid_ratio"] * len(indices))
            num_test = int(test_ratio * len(indices))
            num_train = len(indices) - num_valid - num_test
            train_indices = indices[:num_train]
            valid_indices = indices[num_train : num_train + num_valid]
            test_indices = indices[num_train + num_valid :] if test_ratio > 0 else None

        return train_indices, valid_indices, test_indices


def calculate_class_distribution(masks: list, num_classes: int) -> np.ndarray:
    """
    Calculate class distribution for each mask/label in the dataset.

    Args:
        masks: List of label masks/images
        num_classes: Number of classes in the dataset

    Returns:
        Array of shape (n_samples, num_classes) with normalized class distributions
    """
    distributions = [
        np.histogram(mask, bins=np.arange(num_classes + 1))[0] / mask.size
        for mask in masks
    ]
    return np.array(distributions)


def split_indices_by_clustering(
    masks: list, num_classes: int, n_clusters: int, random_state: int = 42
) -> np.ndarray:
    """
    Split dataset indices into clusters based on label distribution using KMeans.

    Args:
        masks: List of label masks/images
        num_classes: Number of classes in the dataset
        n_clusters: Number of clusters (2 for train/valid, 3 for train/valid/test)
        random_state: Random state for reproducible clustering

    Returns:
        Array of cluster labels for each sample
    """
    distributions = calculate_class_distribution(masks, num_classes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(distributions)
    return kmeans.labels_


def get_label_based_split_indices(
    base_dataset, cfg: dict
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    """
    Get train/valid/test split indices based on label distribution using clustering.

    Args:
        base_dataset: Dataset instance with labels
        cfg: Configuration dictionary

    Returns:
        Tuple of (train_indices, valid_indices, test_indices).
        test_indices is None if test_ratio is 0.
    """
    # Extract all masks/labels from the dataset
    logger.info("Extracting labels for clustering-based split...")
    masks = [base_dataset[i][1] for i in range(len(base_dataset))]

    # Convert masks to numpy arrays if they're tensors
    if isinstance(masks[0], torch.Tensor):
        masks = [mask.cpu().numpy() for mask in masks]

    # Infer number of classes from the data
    all_labels = set()
    for mask in masks:
        all_labels.update(np.unique(mask))
    num_classes = len(all_labels)

    logger.info(f"Found {num_classes} classes for clustering: {sorted(all_labels)}")

    # Determine number of clusters based on test_ratio
    test_ratio = cfg["data"].get("test_ratio", 0.0)
    n_clusters = 3 if test_ratio > 0 else 2

    # Perform clustering
    cluster_labels = split_indices_by_clustering(
        masks, num_classes, n_clusters, random_state=42
    )

    # Convert cluster assignments to indices
    indices = list(range(len(base_dataset)))
    cluster_indices = [[] for _ in range(n_clusters)]

    for idx, cluster_id in enumerate(cluster_labels):
        cluster_indices[cluster_id].append(indices[idx])

    # Stratify within clusters to avoid spatial clustering
    # Instead of assigning entire clusters to splits, we distribute patches
    # from each cluster across train/valid/test to maintain spatial diversity
    valid_ratio = cfg["data"]["valid_ratio"]
    test_ratio = cfg["data"].get("test_ratio", 0.0)

    train_indices = []
    valid_indices = []
    test_indices = [] if test_ratio > 0 else None

    # For each cluster, split its patches according to the ratios
    for cluster_idx in cluster_indices:
        if len(cluster_idx) == 0:
            continue

        # Shuffle cluster indices for random assignment
        cluster_idx_shuffled = cluster_idx.copy()
        np.random.shuffle(cluster_idx_shuffled)

        n_cluster = len(cluster_idx_shuffled)

        if test_ratio > 0:
            # Three-way split
            n_train = int(n_cluster * (1 - valid_ratio - test_ratio))
            n_valid = int(n_cluster * valid_ratio)
            n_test = n_cluster - n_train - n_valid  # remainder goes to test

            train_indices.extend(cluster_idx_shuffled[:n_train])
            valid_indices.extend(cluster_idx_shuffled[n_train : n_train + n_valid])
            if test_indices is not None:  # Type guard
                test_indices.extend(cluster_idx_shuffled[n_train + n_valid :])
        else:
            # Two-way split
            n_train = int(n_cluster * (1 - valid_ratio))
            n_valid = n_cluster - n_train  # remainder goes to validation

            train_indices.extend(cluster_idx_shuffled[:n_train])
            valid_indices.extend(cluster_idx_shuffled[n_train:])

    if test_indices is not None:
        logger.info(
            f"Stratified label-based split: train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}"
        )
    else:
        logger.info(
            f"Stratified label-based split (no test): train={len(train_indices)}, valid={len(valid_indices)}"
        )

    return train_indices, valid_indices, test_indices
