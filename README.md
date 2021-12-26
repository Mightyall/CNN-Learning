# CNN-Learning

We are creating a Convolution Neural Network to test the MNIST


## 12/25/2021 tensorflow implemention for M1

It really takes me time to do that:  https://www.jianshu.com/p/7d27a53e3a5e

Install miniconda with arm64
https://conda-forge.org/blog/posts/2020-10-29-macos-arm64/

Tensorflow Optimizer:
https://github.com/apple/tensorflow_macos



conda create --name envname python=3.8

source activate envname

libs="/Users/kk9912/Desktop/Anaconda/tensorflow_macos/arm64/"


env="/Users/kk9912/miniforge3/envs/envname"


conda upgrade -c conda-forge pip setuptools cached-property six

pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/grpcio-1.33.2-cp38-cp38-macosx_11_0_arm64.whl"

pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/h5py-2.10.0-cp38-cp38-macosx_11_0_arm64.whl"

pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/tensorflow_addons-0.11.2+mlcompute-cp38-cp38-macosx_11_0_arm64.whl"

conda install -c conda-forge -y absl-py
conda install -c conda-forge -y astunparse
conda install -c conda-forge -y gast
conda install -c conda-forge -y opt_einsum
conda install -c conda-forge -y termcolor
conda install -c conda-forge -y typing_extensions
conda install -c conda-forge -y wheel
conda install -c conda-forge -y typeguard

pip install tensorboard

pip install wrapt flatbuffers tensorflow_estimator google_pasta keras_preprocessing protobuf

替换whl文件名字
pip install --upgrade -t "$env/lib/python3.8/site-packages/" --no-dependencies --force "$libs/tensorflow_macos-0.1a1-cp38-cp38-macosx_11_0_arm64.whl"

Now we can try in terminal:
>>python
>>import tensorflow as tf

## Now It Works!

