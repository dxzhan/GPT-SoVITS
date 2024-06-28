#!/bin/bash


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

python -m torch.utils.collect_env