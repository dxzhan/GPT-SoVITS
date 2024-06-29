#!/bin/bash
# ubuntu24.04 RTX2060 CUDA12.2 测试通过
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

sudo apt install ffmpeg
sudo apt install libsox-dev

conda install -c conda-forge gcc
conda install -c conda-forge gxx
#conda install ffmpeg==6.1.1 cmake
conda install ffmpeg==7.0.1 cmake

conda uninstall cuda pytorch torchvision torchaudio pytorch-cuda
#conda install cuda==11.8 pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
#conda install cuda==11.8 pytorch==2.1.2 torchvision==0.16.1 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install cudnn==8.9.2.26 cuda==12.1 cudatoolkit==12.1 pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
#conda install --yes --file requirements.txt

# pip uninstall -r requirements.txt -y
pip install -r requirements.txt

PY_VERSION=`python -V 2>&1|awk '{print $2}'`
PY_VERSION_MAJOR=`python -V 2>&1|awk '{print $2}'|awk -F '.' '{print $1}'`
PY_VERSION_MINOR=`python -V 2>&1|awk '{print $2}'|awk -F '.' '{print $2}'`
#PY_VERSION_REVISION=`python -V 2>&1|awk '{print $2}'|awk -F '.' '{print $3}'`
if [[ "$PY_VERSION_MAJOR" == "3" && "$PY_VERSION_MINOR" == "10" ]]; then
    python -m pip install --upgrade pip
    pip install pyopenjtalk
elif [[ "$PY_VERSION_MAJOR" == "3" && "$PY_VERSION_MINOR" == "12" ]];then
    python -m ensurepip --upgrade
    if [[ ! -d pyopenjtalk ]]; then
        git clone https://github.com/r9y9/pyopenjtalk.git
        git submodule update --recursive --init
    fi
    cd pyopenjtalk    
    pip install -e .
    cd ../
fi

#python -m torch.utils.collect_env