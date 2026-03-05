#!/bin/bash

PYTHON_ENV=/home/bthambiraja/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH
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
echo "activated mdm38_py19"
echo "Running dev/data/process_hdtf/process_hdtf_woz_tracker.py"
python dev/data/process_hdtf/process_hdtf_woz_tracker.py