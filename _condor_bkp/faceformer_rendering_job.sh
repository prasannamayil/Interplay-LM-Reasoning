#!/bin/sh

CONFIG=${1}

PYTHON_ENV=/home/bthambiraja/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH
export PYTHONPATH="${PYTHONPATH}:/home/bthambiraja/projects/NeuralMotionSynthesis"
export HOME="/home/bthambiraja"
export DATA_HOME="/fast/bthambiraja"
export PYOPENGL_PLATFORM="egl"

echo 'START Rendering Job'

source /etc/profile.d/modules.sh
# module load cuda/10.2

echo 'ACTIVATE CONDA faceformer'
source ${PYTHON_ENV}
conda activate faceformer

echo 'RUN SCRIPT'
echo 'ScriptDir' ${SCRIPT_DIR}
echo 'CONFIG: ' ${CONFIG}
cd ${SCRIPT_DIR}/..
echo "Script executed from: ${PWD}"
# python ./dev/result_helpers/generate_heat_map_for_sequence.py --model ${CONFIG}
python ./dev/result_helpers/generate_gt_vid.py