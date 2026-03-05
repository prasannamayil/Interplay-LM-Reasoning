#!/bin/sh
export HOME="/home/bthambiraja"
export DATA_HOME="/work/bthambiraja"
export PYOPENGL_PLATFORM="egl"
export LOGHOME="."
export PYTHONPATH=$PYTHONPATH:/home/bthambiraja/projects/motion_root
export PYTHONPATH=$PYTHONPATH:/home/bthambiraja/projects/motion_root/motion_diffusion_model:/home/bthambiraja/projects/motion_root/face_animation_model

source /etc/profile.d/modules.sh
module load cuda/11.6

echo 'ACTIVATE CONDA mdm38_py19'
source ${PYTHON_ENV}
conda activate mdm38_py19

nvidia-smi