#!/bin/sh

sequence=${1}
POSTFIX=${2:-def}

PYTHON_ENV=/home/bthambiraja/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH
export HOME="/home/bthambiraja"
export DATA_HOME="/fast/bthambiraja"
export PYOPENGL_PLATFORM="egl"

echo 'START JOB (faceformer training)'

source /etc/profile.d/modules.sh
module load cuda/10.2

echo 'ACTIVATE CONDA faceformer'
source ${PYTHON_ENV}
conda activate faceformer

nvidia-smi

# echo 'RUN SCRIPT'
echo 'ScriptDir' ${SCRIPT_DIR}
echo 'sequence: ' ${sequence}
cd ${SCRIPT_DIR}/..
echo "Script executed from: ${PWD}"

dataset_path=/is/rg/ncs/datasets/ARDZDF
ffmpeg -y -i ${dataset_path}/${sequence}/input_video.mp4 -vf fps=30 ${dataset_path}/${sequence}/images_png_30fps/frame%04d.png