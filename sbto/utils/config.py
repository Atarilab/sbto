from dataclasses import dataclass, asdict, field
from abc import abstractmethod
import yaml
import os
import numpy as np

@dataclass
class ConfigAbstract:
    _filename: str = field(init=False, repr=False)

    @abstractmethod
    def save(self, dir_path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        pass

    @property
    def args(self):
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}
    
    def set_filename(self, filename: str) -> None:
        self._filename = filename
    
@dataclass
class ConfigBase(ConfigAbstract):

    def __post_init__(self):
        self._filename = "config"

    def save(self, dir_path: str):
        """Implements saving to YAML."""
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, self._filename)
        with open(file_path, "w") as f:
            yaml.safe_dump(self.args, f, sort_keys=False)
        print(f"Config saved to {file_path}")

    @classmethod
    def load(cls, path: str):
        """Implements loading from YAML."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @property
    def args(self):
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}
    
@dataclass
class ConfigNPZBase(ConfigAbstract):

    def __post_init__(self):
        self._filename = "config.npz"

    def save(self, dir_path: str):
        """Save configuration parameters to an .npz file."""
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, self._filename)

        # Convert all arguments to serializable numpy-compatible types
        args = {}
        for k, v in self.args.items():
            if isinstance(v, (list, tuple)):
                args[k] = np.array(v)
            else:
                args[k] = v

        np.savez(file_path, **args)
        # print(f"Config saved to {file_path}")

    @classmethod
    def load(cls, path: str):
        """Load configuration parameters from an .npz file."""
        data = np.load(path, allow_pickle=True)
        kwargs = {key: data[key].tolist() if data[key].ndim == 0 else data[key] for key in data.files}
        return cls(**kwargs)

    @property
    def args(self):
        """Return a dictionary of configuration fields, excluding private ones."""
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}