"""
Shared Training Modules
=======================

Dit package bevat gedeelde functionaliteit voor alle training scripts:
- Argument parsing
- Data loading
- Environment setup
- Logging setup

Door deze code te delen vermijden we duplicatie en zorgen we voor
consistentie tussen PPO en SAC training scripts.

Auteur: DataDeepRL Team
"""

from train.common.args import (
    add_data_args,
    add_training_args,
    add_env_args,
    add_logging_args,
    add_model_args
)

from train.common.setup import (
    setup_device,
    setup_seed,
    load_data,
    load_coredata,
    STATIONARY_FEATURES,
    create_environments,
    setup_logger
)

__all__ = [
    'add_data_args',
    'add_training_args', 
    'add_env_args',
    'add_logging_args',
    'add_model_args',
    'setup_device',
    'setup_seed',
    'load_data',
    'load_coredata',
    'STATIONARY_FEATURES',
    'create_environments',
    'setup_logger'
]
