from dataclasses import dataclass, asdict
import yaml
import os

@dataclass
class ConfigBase:

    def save_to_yaml(self, dir_path: str):
        """Implements saving to YAML."""
        FILENAME = "config.yaml"
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, FILENAME)
        with open(file_path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
        print(f"Config saved to {file_path}")

    @classmethod
    def load_from_yaml(cls, path: str):
        """Implements loading from YAML."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @property
    def args(self):
        return asdict(self)