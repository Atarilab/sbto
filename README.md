# Sampling Based Trajectory Optimization (SBTO)

This repository contains the official implementation of the SBTO mentioned in the paper:

> **DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization**
> Victor Dhedin, Ilyass Taouil, Shafeef Omar, Dian Yu, Kun Tao, Angela Dai, Majid Khadiv
> arXiv:2602.06827 · [Paper](https://arxiv.org/abs/2602.06827)

DynaRetarget is a complete pipeline for retargeting human motions to humanoid control policies. The core component is a novel Sampling-Based Trajectory Optimization (SBTO) framework that refines imperfect kinematic trajectories into dynamically feasible motions. SBTO incrementally advances the optimization horizon, enabling optimization over the entire trajectory for long-horizon tasks. The framework generalizes across varying object properties such as mass, size, and geometry using the same tracking objective.

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
#### Environment
```bash
https://github.com/Atarilab/sbto.git
cd sbto
conda create -n sbto python=3.12.11
conda activate sbto
pip install --upgrade pip mujoco==3.3.7 numba==0.62.1 scipy==1.16.2 matplotlib==3.10.6 pyyaml==6.0.3 hydra-core==1.3.2 seaborn==0.13.2
conda install -c conda-forge opencv
pip install -e .
```

#### OmniRetarget
Download robot-object motion references from Omniretarget dataset.
```bash
mkdir datasets && cd datasets
wget "https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/resolve/main/robot-object.zip"
unzip robot-object.zip
```

## Usage
Most of the paramters of SBTO can be set at runtime as command line argument. The code base relies on [hydra](https://hydra.cc/) to do so. Parameters required to instantiate the different classes in the code can be found in the `./conf` sub-directories.
For more advance usage (if for instance you want to write your own task/solvers), I recommend looking into the different config files to have a better understanding of the repo structure.

### Loading a motion reference
To run SBTO on a specific motion reference from the OmniRetarget dataset simply run:
```python
python3 sbto/main.py \
# Change the solver (cem is the default one)
solver=cem \
task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz
```
One can have more control on the motion reference by changing the parameters defined in the respective config [file](sbto/conf/task/g1/cfg_ref/default.yaml).

**Warning**: If you use your own reference motion in MuJoCo format then you should set `task.cfg_ref.flip_quat_pos=False`. This is set to True by default as for OmniRetarget data, free joints are expressed in [quat, pos] format.

To check that your reference is being loaded correctly, you can visualize it by running:
```python
python3 scripts/visualize_ref.py \
task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
task.cfg_ref.speedup=2.
# Add all hydra args you would use for SBTO
```

### Changing the scene
SBTO also allows to change the scene directly from command line arguments.

Very importantly, SBTO loads **two different scenes** when using a reference: the one of the demonstration and the one of the refinement process (in which the rollouts happen).

Predefined scenes are already defined [here](sbto/conf/task/g1/mj_scene_ref) (for the reference) and [here](sbto/conf/task/g1/sim/mj_scene) (for the rollouts).

For the OmniRetarget dataset, the reference is a box. For the rollouts one can use different options with different objects:
```python
python3 sbto/main.py \
solver=cem \
task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
# Here the hydra command gets a bit heavy \
task/g1/sim/mj_scene@task.sim.mj_scene=small_box  # can be chair, shelf, cylinder
```

If you want to add your own objects, SBTO supports primitive geometries, `.urdf` and `.obj` meshes. Note the object placement has to be manually refined so that it starts in the correct position and orientation.

If you want to visualize your scene, you can change the reference's scene and use the same script as before:
```python3
python3 scripts/visualize_ref.py \
task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
task/g1/mj_scene_ref@task.mj_scene_ref=../sim/mj_scene/chair_mesh
```

#### Without object
If you don't have any object in your scene use `g1/robot_ref` task:
```python3
python3 sbto/main.py \
task=g1/robot_ref \
task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz
```

## Citation
If you use this code in your research, please cite:
```bibtex
@article{dhedin2025dynaretarget,
  title     = {DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization},
  author    = {Dhedin, Victor and Taouil, Ilyass and Omar, Shafeef and Yu, Dian and Tao, Kun and Dai, Angela and Khadiv, Majid},
  journal   = {arXiv preprint arXiv:2602.06827},
  year      = {2025}
}
```
