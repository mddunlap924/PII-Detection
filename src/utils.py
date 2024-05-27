"""
Miscellaneous and helper code for various tasks will be used in this script.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import uuid
import random
import yaml
import numpy as np
import torch


class RecursiveNamespace(SimpleNamespace):
    """
    Extending SimpleNamespace for Nested Dictionaries
    # https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8

    Args:
        SimpleNamespace (_type_): Base class is SimpleNamespace

    Returns:
        _type_: A simple class for nested dictionaries
    """
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, RecursiveNamespace(**val))
            elif isinstance(val, list):
                setattr(self, key, list(map(self.map_entry, val)))

    def to_dict(self):
        def convert(obj):
            if isinstance(obj, RecursiveNamespace):
                return {key: convert(value) for key, value in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        return convert(self)


def load_cfg(base_dir: Path,
             filename: str,
             *,
             as_namespace: bool = True) -> SimpleNamespace:
    """
    Load YAML configuration files saved uding the "cfgs" directory
    Args:
        base_dir (Path): Directory to YAML config. file
        filename (str): Name of YAML configuration file to load
    Returns:
        SimpleNamespace: A simple class for calling configuration parameters
    """
    cfg_path = Path(base_dir) / filename
    with open(cfg_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    file.close()
    if as_namespace:
        cfg = RecursiveNamespace(**cfg_dict)
    else:
        cfg = cfg_dict
    return cfg


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def seed_everything(*, seed: int = 42):
    """
    Seed everything

    Args:
        seed (_type_): Seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class RunIDs():
    def __init__(self,
                 test_folds: list,
                 num_folds: int,
                 save_dir: str,
                 save_results: bool):
        self.test_folds = test_folds
        self.num_folds = num_folds
        self.save_dir = Path(save_dir)
        self.save_results = save_results
        self.group_id = str
        self.folds_id = RecursiveNamespace

        # Are all folds being tested
        if test_folds == list(range(num_folds)):
            self.test_all_folds = True
        else:
            self.test_all_folds = False

        # Check if base directory for saving results exists
        if (not Path(save_dir).exists()) and save_results:
            raise Exception('Base Dir. for Saving Results Does NOT Exist')

    def generate_id(self):
        # Generate a random ID
        return str(uuid.uuid4()).split('-')[0]

    def generate_run_ids(self):
        # Get a group id (i.e. ID that will organize all folds)
        self.group_id = self.generate_id()

        # Info. for each data fold
        fold_info = {}
        for fold in self.test_folds:
            # Create directory for each fold
            fold_dir = self.save_dir / self.group_id / \
                f'fold{fold}of{self.num_folds}'
            if self.save_results:
                fold_dir.mkdir(parents=True)
                save_path = fold_dir
            else:
                save_path = None

            # Store info
            fold_info[f'fold{fold}'] = {'run_id': self.generate_id(),
                                        'save': self.save_results,
                                        'path': save_path,
                                        'fold_num': fold}
        self.folds_id = RecursiveNamespace(**fold_info)
        return
