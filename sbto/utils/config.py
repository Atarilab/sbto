from dataclasses import dataclass, asdict, field
import yaml
import os

@dataclass
class ConfigBase:
    _filename: str = field(init=False, repr=False)

    def __post_init__(self):
        self._filename = "config.yaml"

    def save_to_yaml(self, dir_path: str):
        """Implements saving to YAML."""
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, self._filename)
        with open(file_path, "w") as f:
            yaml.safe_dump(self.args, f, sort_keys=False)
        print(f"Config saved to {file_path}")

    @classmethod
    def load_from_yaml(cls, path: str):
        """Implements loading from YAML."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @property
    def args(self):
        return {k: v for k, v in asdict(self).items() if k != "_filename"}