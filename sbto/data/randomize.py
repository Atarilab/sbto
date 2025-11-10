import numpy as np
from typing import List, Tuple, Any, Union
import numpy as np
from copy import copy

from sbto.utils.config import ConfigBase

class RandomizedParamConfig(ConfigBase):
    def __post_init__(self):
        self._filename = "data_randomization"
        self.name2kwargs = {}

    def randomize(
        self,
        name: str,
        range: Tuple,
        axis: int | tuple = 0,
        logscale: bool = False
        ) -> None:
        self.name2kwargs[name] = {
            "range":range,
            "axis":axis,
            "logscale":logscale,
        }

def _randomize_param(
    config: ConfigBase,
    name: str,
    range: Tuple,
    num: int,
    axis: int | tuple = 0,
    logscale: bool = False
    ) -> Any:
    """
    Get <num> randomized value of a parameter within a <range>.
    """
    if not name in config.args:
        raise ValueError(f"Parameter {name} not found in config.")

    param = config.args[name]
    arr_param = np.asarray(param)

    has_multiple_dim = np.sum(arr_param.shape) > 1
    is_tuple = isinstance(param, tuple)
    is_list = isinstance(param, list)
    is_float = isinstance(param, float)
    is_int = isinstance(param, int)
    if not any((is_float, is_int, is_tuple, is_list)):
        raise ValueError(f"Parameter {name} should be a float, int, list, or tuple.")

    start, stop = range
    # --- random sampling ---
    if not logscale:
        rand_values = np.random.uniform(start, stop, num)
    else:
        rand_values = np.exp(np.random.uniform(np.log(start), np.log(stop), num))

    # Cast to the right type
    def cast_type(v):
        if is_int:
            v = int(round(v))
        elif is_float:
            v = float(v)
        elif is_tuple:
            arr = arr_param.copy()
            if has_multiple_dim:
                arr[axis] = v
            else:
                arr = np.array([v])
            v = tuple(arr.tolist())
        elif is_list:
            arr = arr_param.copy()
            if has_multiple_dim:
                arr[axis] = v
            else:
                arr = np.array([v])
            v = arr.tolist()
        return v

    casted_rand_values = list(map(cast_type, rand_values))
    return casted_rand_values

def get_randomized_config(
    config: ConfigBase,
    config_param: RandomizedParamConfig,
    num: int,
    ) -> List[ConfigBase]:
    """
    Returns a list of config objects with the given parameter randomly
    sampled within the specified range.
    """
    configs = [copy(config) for _ in range(num)]

    # Get randomized values for each param
    for name, kwargs in config_param.name2kwargs.items():
        randomized_values = _randomize_param(
            config=config,
            name=name,
            num=num,
            **kwargs
        )
        # Set values for each config
        for i, v in enumerate(randomized_values):
            setattr(configs[i], name, v)
    
    return configs
