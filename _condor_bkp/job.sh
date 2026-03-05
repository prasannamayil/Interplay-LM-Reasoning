#!/bin/sh

CONFIG=${1}
POSTFIX=${2:-def}

PYTHON_ENV=/home/bthambiraja/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH
export HOME="/home/bthambiraja"
export DATA_HOME="/fast/bthambiraja"
export PYOPENGL_PLATFORM="egl"

echo 'START JOB (VQGAN training)'

source /etc/profile.d/modules.sh
module load cuda/10.2

echo 'ACTIVATE CONDA TAMING_V2'
source ${PYTHON_ENV}
conda activate taming_v2

nvidia-smi

echo 'RUN SCRIPT'
echo 'ScriptDir' ${SCRIPT_DIR}
echo 'CONFIG: ' ${CONFIG}
cd ${SCRIPT_DIR}/..
echo "Script executed from: ${PWD}"
python ./main.py -b ${CONFIG} --train --gpus 0, --postfix ${POSTFIX}