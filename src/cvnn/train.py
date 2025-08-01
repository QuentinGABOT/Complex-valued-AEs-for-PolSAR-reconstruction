# Standard library imports
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import torch
import torchcvnn.nn.modules as c_nn  # for loss classes
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from tqdm import tqdm

# Local imports
from cvnn.callbacks import ModelCheckpoint
from cvnn.models.utils import get_loss_function
from cvnn.schedulers import build_schedulers, step_schedulers
from cvnn.utils import setup_logging

# initialize module-level logger
logger = setup_logging(__name__)


def generate_unique_logpath(logdir: Union[str, Path], raw_run_name: str) -> str:
    """
    Create a new run folder under `logdir` named `raw_run_name_N` where N is one greater than existing.

    Arguments:
        logdir: the prefix directory
        raw_run_name: the base name

    Returns:
        log_path: a non-existent path like logdir/raw_run_name_x
                  where x is an int that is higher than any existing suffix.
    """
    # create new numbered run folder
    base = Path(logdir)
    nums = [
        int(p.name.rsplit("_", 1)[1])
        for p in base.iterdir()
        if p.is_dir()
        and p.name.startswith(f"{raw_run_name}_")
        and p.name.rsplit("_", 1)[1].isdigit()
    ]
    next_idx = max(nums) + 1 if nums else 0
    new_dir = base / f"{raw_run_name}_{next_idx}"
    new_dir.mkdir()
    return str(new_dir)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_scheduler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    sched_step_on_batch: bool = False,
) -> float:
    """Run one training epoch and return average loss, optionally stepping schedulers per batch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in tqdm(train_loader, desc="Training"):  # progress bar
        # Handle both reconstruction (single tensor) and segmentation (input, target pairs)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            # Segmentation task: use provided targets
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Convert targets to LongTensor for classification losses
            if targets.dtype == torch.uint8:
                targets = targets.long()
        elif isinstance(batch, (tuple, list)) and len(batch) == 1:
            # Single tensor wrapped in tuple (from TensorDataset)
            inputs = batch[0]
            inputs = inputs.to(device)
            targets = inputs
            # For reconstruction tasks, use processed inputs as targets
        else:
            # Direct tensor
            inputs = batch
            inputs = inputs.to(device)
            targets = inputs
            # For reconstruction tasks, use processed inputs as targets

        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            # If model returns tuple, use the projected outputs
            # This is the intended behavior in models with projection
            outputs_not_projected = outputs[0]
            outputs = outputs[1]
        loss = loss_fn(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        # batch-level warmup stepping
        if warmup_scheduler:
            warmup_scheduler.step()
        # batch-level scheduler stepping
        if (
            scheduler
            and sched_step_on_batch
            and not isinstance(scheduler, ReduceLROnPlateau)
        ):
            scheduler.step()
        bs = inputs.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
    return total_loss / total_samples


def validate_one_epoch(
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    """Run one validation epoch and return average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation"):  # progress bar
            # Handle both reconstruction (single tensor) and segmentation (input, target pairs)
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                # Segmentation task: use provided targets
                inputs, targets = batch[0], batch[1]
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Convert targets to LongTensor for classification losses
                if targets.dtype == torch.uint8:
                    targets = targets.long()
            elif isinstance(batch, (tuple, list)) and len(batch) == 1:
                # Single tensor wrapped in tuple (from TensorDataset)
                inputs = batch[0]
                inputs = inputs.to(device)
                targets = inputs
            else:
                # Direct tensor
                inputs = batch
                inputs = inputs.to(device)
                targets = inputs

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                # If model returns tuple, use the projected outputs
                # This is the intended behavior in models with projection
                outputs_not_projected = outputs[0]
                outputs = outputs[1]
            loss = loss_fn(outputs, targets)
            bs = inputs.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
    return total_loss / total_samples


def setup_loss_optimizer(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
) -> Tuple[Callable, torch.optim.Optimizer]:
    """Instantiate loss function and optimizer based on config."""
    loss_name = cfg["loss"]["name"]

    # Get the model's layer_mode for mode-aware loss selection
    layer_mode = getattr(model, "layer_mode", "complex")

    # Get ignore_index from config if available
    ignore_index = cfg.get("training", {}).get("ignore_index", None)

    # Try to use mode-aware loss selection first
    try:
        loss_fn = get_loss_function(loss_name, layer_mode, ignore_index)
    except ValueError:
        # Fall back to old behavior for backward compatibility
        try:
            loss_cls = getattr(c_nn, loss_name)
        except AttributeError:
            import torch.nn as nn

            loss_cls = getattr(nn, loss_name)

        # Apply ignore_index for backward compatibility fallback if it's CrossEntropy
        if ignore_index is not None and loss_name == "CrossEntropyLoss":
            loss_fn = loss_cls(ignore_index=ignore_index)
        else:
            loss_fn = loss_cls()

    optim_cls = getattr(torch.optim, cfg["optim"]["algo"])
    optimizer = optim_cls(model.parameters(), **cfg["optim"]["params"])
    return loss_fn, optimizer


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    logdir: Union[str, Path],
    device: Optional[torch.device] = None,
    loss_fn: Optional[Callable] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    warmup_scheduler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    start_epoch: int = 0,
    gumbel_experiment: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """Run full training over epochs and return history of losses.

    Args:
        start_epoch: Starting epoch for training (0 for new training, >0 for retrain)
        gumbel_experiment: Optional BaseExperiment instance for Gumbel tau handling
    """
    # default loss/optimizer if omitted
    default_loss_fn, default_optimizer = setup_loss_optimizer(model, cfg)
    if loss_fn is None:
        loss_fn = default_loss_fn
    if optimizer is None:
        optimizer = default_optimizer

    if device is None:
        device = next(model.parameters()).device

    # logging configuration
    logger.info(f"Config: {cfg}")

    # build warmup and main schedulers if not provided
    if warmup_scheduler is None or scheduler is None:
        w, s = build_schedulers(optimizer, cfg, len(train_loader))
        warmup_scheduler = warmup_scheduler or w
        scheduler = scheduler or s
    # determine scheduler stepping granularity
    sched_cfg = cfg.get("scheduler", {})
    step_on_batch = sched_cfg.get("step_on", "epoch") == "batch"
    # initialize checkpoint
    # determine input dimensions for export
    sample_batch = next(iter(train_loader))
    sample = (
        sample_batch[0] if isinstance(sample_batch, (tuple, list)) else sample_batch
    )
    num_input_dims = sample.ndim
    
    # Use custom checkpoint if gumbel_experiment is provided
    if gumbel_experiment is not None:
        checkpoint = ModelCheckpointWithGumbel(
            model,
            optimizer,
            logdir,
            num_input_dims,
            min_is_best=True,
            warmup_scheduler=warmup_scheduler,
            scheduler=scheduler,
            gumbel_experiment=gumbel_experiment,
        )
    else:
        checkpoint = ModelCheckpoint(
            model,
            optimizer,
            logdir,
            num_input_dims,
            min_is_best=True,
            warmup_scheduler=warmup_scheduler,
            scheduler=scheduler,
        )
    history = {"train_loss": [], "valid_loss": []}

    # Calculate actual epoch range for training
    total_epochs = cfg["nepochs"]
    if start_epoch >= total_epochs:
        logger.warning(
            f"Start epoch {start_epoch} >= total epochs {total_epochs}, no training needed"
        )
        return history

    logger.info(f"Training from epoch {start_epoch} to {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        # Update Gumbel tau if experiment is provided
        if gumbel_experiment is not None:
            gumbel_experiment.current_gumbel_tau = update_gumbel_tau(
                model, 
                gumbel_experiment.gumbel_tau_config, 
                gumbel_experiment.current_gumbel_tau, 
                epoch
            )
            
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            warmup_scheduler,
            scheduler,
            step_on_batch,
        )
        valid_loss = validate_one_epoch(model, valid_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        
        # Log to wandb if initialized
        if wandb.run:
            log_dict = {
                "training/loss": train_loss,
                "validation/loss": valid_loss,
                "epoch": epoch,
            }
            
            # Add learning rate
            current_lr = optimizer.param_groups[0]['lr']
            log_dict["training/learning_rate"] = current_lr
            
            # Add gumbel tau if available
            if gumbel_experiment and hasattr(gumbel_experiment, 'current_gumbel_tau'):
                log_dict["training/gumbel_tau"] = gumbel_experiment.current_gumbel_tau.item()
                
            wandb.log(log_dict)
            
        # update checkpoint with validation score
        checkpoint.update(valid_loss, epoch)
        
        # save a snapshot after each epoch for crash recovery
        last_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if warmup_scheduler:
            last_state["warmup_state_dict"] = warmup_scheduler.state_dict()
        if scheduler:
            last_state["scheduler_state_dict"] = scheduler.state_dict()
            
        # Add Gumbel tau state if experiment is provided
        if gumbel_experiment is not None:
            gumbel_state = gumbel_experiment.get_gumbel_tau_state()
            if gumbel_state:
                last_state["gumbel_tau_state"] = gumbel_state
                
            # Add projection state if experiment has it
            if hasattr(gumbel_experiment, 'get_projection_state'):
                projection_state = gumbel_experiment.get_projection_state()
                if projection_state:
                    last_state["projection_state"] = projection_state
                
        torch.save(last_state, Path(logdir) / "last_model.pt")
        
        # step schedulers (only main scheduler at epoch; warmup already stepped in batches)
        step_schedulers(None, scheduler, metric=valid_loss, on_batch=False)
    return history


def update_gumbel_tau(model: torch.nn.Module, gumbel_config: Dict[str, Any], current_tau: torch.Tensor, epoch: int) -> torch.Tensor:
    """Update Gumbel tau based on epoch and decay schedule.
    
    Args:
        model: The model containing encoder blocks with Gumbel tau
        gumbel_config: Configuration dict with start_value, gamma, start_decay_epoch, min_value
        current_tau: Current tau value
        epoch: Current training epoch
        
    Returns:
        Updated tau value
    """
    if epoch >= gumbel_config["start_decay_epoch"]:
        decay_epochs = epoch - gumbel_config["start_decay_epoch"]
        new_tau_value = gumbel_config["start_value"] * (gumbel_config["gamma"] ** decay_epochs)
        new_tau = torch.tensor(
            max(new_tau_value, gumbel_config["min_value"]), 
            dtype=torch.float32
        )
        
        # Update all encoder blocks
        if hasattr(model, 'encoder'):
            for enc in model.convnet.encoder[1:]:
                if hasattr(enc, 'downsampling_method') and hasattr(enc.down, 'component_selection'):
                    enc.down.component_selection.gumbel_tau = new_tau
        
        return new_tau
    
    return current_tau


class ModelCheckpointWithGumbel(ModelCheckpoint):
    """Extended ModelCheckpoint that also saves Gumbel tau state."""
    
    def __init__(self, *args, gumbel_experiment=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gumbel_experiment = gumbel_experiment
    
    def update(self, score: float, epoch: int) -> bool:
        """Update checkpoint if score improved, including Gumbel tau state."""
        if self.is_better(score):
            self.model.eval()
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": score,
            }
            if self.warmup_scheduler is not None:
                state["warmup_state_dict"] = self.warmup_scheduler.state_dict()
            if self.scheduler is not None:
                state["scheduler_state_dict"] = self.scheduler.state_dict()
            
            # Add Gumbel tau state if experiment has it
            if self.gumbel_experiment and hasattr(self.gumbel_experiment, 'get_gumbel_tau_state'):
                gumbel_state = self.gumbel_experiment.get_gumbel_tau_state()
                if gumbel_state:
                    state["gumbel_tau_state"] = gumbel_state
                    
                # Add projection state if experiment has it
                if hasattr(self.gumbel_experiment, 'get_projection_state'):
                    projection_state = self.gumbel_experiment.get_projection_state()
                    if projection_state:
                        state["projection_state"] = projection_state
            
            torch.save(state, self.savepath / "best_model.pt")
            self.best_score = score
            return True
        return False
