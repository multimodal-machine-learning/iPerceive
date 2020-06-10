#!/bin/bash

# # this installs the right pip and dependencies for the fresh python

# CUDA Toolkit 10.2 Download
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

#conda install python=3.7
# maskrcnn_benchmark and coco api dependencies
# install latest versions in your global environment
pip3 install ninja yacs cython matplotlib tqdm opencv-python h5py lmdb

# for maskrcnn
conda install tensorflow scikit-image
conda remove imgaug
conda install -c conda-forge geos
conda install -c conda-forge shapely
conda config --add channels conda-forge
conda install imgaug

# install specific versions in your conda environment
# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=10.0

# install pytorch
#conda install -c pytorch pytorch-nightly=1.0.0 torchvision=0.2.2
pip3 install 'pillow<7.0.0'

# install pycocotools
cd ..
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python3 setup.py build_ext install

# install apex
cd ..
cd ..
git clone https://github.com/NVIDIA/apex.git
cd apex
python3 setup.py install --cuda_ext --cpp_ext
cd ..

# install VC
cd VC-R-CNN
python3 setup.py build develop

