# SBTO
Sampling Based Trajectory Optimization (SBTO)

## Dependencies

- python=3.12.11
- numpy=2.3.4
- mujoco=3.3.7
- numba=0.62.1
- scipy=1.16.2
- matplotlib=3.10.6
- pyyaml=6.0.3
- opencv-python=4.12.0
- hydra-core==1.3.2

### Install
```bash
https://github.com/Atarilab/sbto.git
cd sbto
conda create -n sbto python=3.12.11
conda activate sbto
pip install --upgrade pip mujoco==3.3.7 numba==0.62.1 scipy==1.16.2 matplotlib==3.10.6 pyyaml==6.0.3 hydra==1.3.2
conda install -c conda-forge opencv
pip install -e .
```

### Test
```python
python3 ./exemples/g1_gait.py
```