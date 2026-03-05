#!/bin/bash

export SHARED_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
export PROJECT_ROOT="/home/bthambiraja/projects/Interplay-LM-Reasoning"
export VENV_PATH="/fast/bthambiraja/projects/Interplay-LM-Reasoning"

conda deactivate
source /etc/profile.d/modules.sh

module load cuda/12.1
module load cudnn/8.9.1-cu12.x

python3 -m venv $VENV_PATH/gsm_pretrain
source $VENV_PATH/gsm_pretrain/bin/activate
pip install --upgrade pip setuptools wheel

## setuping up pip
pip install -r $PROJECT_ROOT/requirements.txt

# dLLM (diffusion language modeling)
cd $PROJECT_ROOT/dllm
pip install -e .

# LLaMA-Factory (transformer baselines)
cd $PROJECT_ROOT/LLaMA-Factory
pip install -e ".[torch,deepspeed,metrics]"

# gsm_infinite (data generation / evaluation benchmark)
cd $PROJECT_ROOT/gsm_infinite
pip install -e .

pip install flash-attn --no-build-isolation


python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
python -c "import dllm; print('dllm OK')"
python -c "import transformers; print('transformers OK')"
python -c "import llamafactory; print('LLaMA-Factory OK')"