# Standard library imports
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party imports
import numpy as np
import torch
import yaml

# Local imports
from cvnn.config import load_config


def set_seed(seed: int) -> None:
    """
    Set seed for reproducible results across random, numpy, and torch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    name: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Return a logger with StreamHandler and formatter if not already configured.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger
