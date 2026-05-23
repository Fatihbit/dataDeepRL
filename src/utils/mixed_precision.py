"""
Mixed Precision Training Utilities
==================================

Dit module biedt utilities voor mixed precision training met PyTorch's
automatic mixed precision (AMP). Mixed precision kan training significant
versnellen op moderne GPUs (Volta, Turing, Ampere) met minimaal accuratieverlies.

Wat is Mixed Precision?
-----------------------
- Gebruikt FP16 (half precision) voor forward/backward passes
- Houdt FP32 (full precision) kopie van weights voor updates
- Automatische loss scaling om underflow te voorkomen

Voordelen:
- 2-3x snellere training op compatible GPUs
- 50% minder GPU memory gebruik
- Geen significante accuracy degradatie

Gebruik:
    >>> trainer = MixedPrecisionTrainer(enabled=True)
    >>> with trainer.autocast():
    ...     loss = model(inputs)
    >>> trainer.backward(loss, optimizer)

Auteur: DataDeepRL Team
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class MixedPrecisionTrainer:
    """
    Helper class voor mixed precision training.
    
    Beheert automatisch:
    - AMP autocast context voor forward passes
    - GradScaler voor loss scaling en gradient unscaling
    - Graceful fallback naar FP32 als AMP niet beschikbaar is
    
    Args:
        enabled: Of mixed precision aan staat (default: False voor backward compat)
        init_scale: Initiële loss scale factor
        growth_factor: Factor waarmee scale groeit na succesvolle steps
        backoff_factor: Factor waarmee scale krimpt na inf/nan
        growth_interval: Steps tussen scale growth attempts
        
    Voorbeeld:
        >>> trainer = MixedPrecisionTrainer(enabled=True)
        >>> 
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     
        ...     # Forward pass met autocast
        ...     with trainer.autocast():
        ...         outputs = model(batch)
        ...         loss = criterion(outputs, targets)
        ...     
        ...     # Backward met scaling
        ...     trainer.backward(loss, optimizer)
        ...     
        ...     # Optimizer step met unscaling
        ...     trainer.step(optimizer)
    """
    
    def __init__(
        self,
        enabled: bool = False,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        self.enabled = enabled and torch.cuda.is_available()
        
        if self.enabled:
            # Check of GPU mixed precision ondersteunt
            if not self._check_amp_support():
                logger.warning("GPU does not fully support mixed precision. Disabling AMP.")
                self.enabled = False
        
        # Initialiseer GradScaler alleen als enabled
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=True
            )
            logger.info("Mixed precision training ENABLED")
        else:
            self.scaler = None
            if enabled:
                logger.info("Mixed precision requested but CUDA not available, using FP32")
    
    def _check_amp_support(self) -> bool:
        """Check of de GPU mixed precision ondersteunt."""
        if not torch.cuda.is_available():
            return False
        
        # Check compute capability (7.0+ = Volta, Turing, Ampere)
        capability = torch.cuda.get_device_capability()
        # Compute capability 7.0+ heeft native FP16 tensor cores
        return capability[0] >= 7
    
    @contextmanager
    def autocast(self, dtype: torch.dtype = torch.float16):
        """
        Context manager voor automatic mixed precision.
        
        Args:
            dtype: Data type voor AMP (float16 of bfloat16)
            
        Yields:
            Context voor forward pass met mixed precision
        """
        if self.enabled:
            with autocast(dtype=dtype):
                yield
        else:
            yield
    
    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, retain_graph: bool = False):
        """
        Backward pass met loss scaling.
        
        Args:
            loss: Loss tensor om te backpropageren
            optimizer: Optimizer (nodig voor gradient unscaling)
            retain_graph: Of computation graph behouden moet worden
        """
        if self.enabled and self.scaler is not None:
            # Scale loss en backward
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            # Normale backward
            loss.backward(retain_graph=retain_graph)
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Optimizer step met gradient unscaling.
        
        Args:
            optimizer: Optimizer om te steppen
        """
        if self.enabled and self.scaler is not None:
            # Unscale gradients en check voor inf/nan
            self.scaler.step(optimizer)
            # Update scale factor
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients voor gradient clipping.
        
        Roep dit aan VOOR gradient clipping en VOOR optimizer.step().
        
        Args:
            optimizer: Optimizer waarvan gradients unscaled moeten worden
        """
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def get_scale(self) -> float:
        """Krijg huidige loss scale factor."""
        if self.enabled and self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> dict:
        """Krijg state dict voor checkpointing."""
        if self.enabled and self.scaler is not None:
            return {'scaler': self.scaler.state_dict(), 'enabled': self.enabled}
        return {'enabled': self.enabled}
    
    def load_state_dict(self, state_dict: dict):
        """Laad state dict van checkpoint."""
        if 'scaler' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])


def get_autocast_context(enabled: bool = False, dtype: torch.dtype = torch.float16):
    """
    Get autocast context manager.
    
    Convenience functie voor simpele use cases.
    
    Args:
        enabled: Of mixed precision aan staat
        dtype: Data type (float16 of bfloat16)
        
    Returns:
        Context manager voor autocast
    """
    if enabled and torch.cuda.is_available():
        return autocast(dtype=dtype)
    else:
        return _nullcontext()


class _nullcontext:
    """Null context manager voor fallback."""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
