# Napari-Chloroplasts

## Installation
```bash
conda create -n chlo 'python==3.10.12' pytorch
# if using GPU
# conda create -n chlo 'python==3.10.12' pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate chlo
# install omnipose and napari
pip install omnipose
pip install 'napari[all]'
# clone this repository and install
mkdir <where/you/want/this/repo/to/be> # e.g. ~/code/
cd <where/you/want/this/repo/to/be>
git clone git@github.com:niwaka-ame/napari-chloroplasts.git
cd <path/to/this/repo>
pip install -e .
# if there's an import issue from omnipose (for GPU only)
# conda install "mkl<2024.1"
```

