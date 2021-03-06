#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda-8-0 -y
fi

tar xvzf ~/tensorflowCode/linux_setup/cudnn-8.0-linux-x64-v5.1.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
printf '\nexport LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc
printf '\nexport CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

printf 'Y\n' | sudo apt-get install libcupti-dev
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export LD_LIBRARY_PATH="$/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"

export CUDA_HOME=/usr/local/cuda

wget -N https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash ./Anaconda3-4.4.0-Linux-x86_64.sh -b -p ~/anaconda3

printf '\nexport PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH

printf 'y' | conda create -n tensorflowGPU python=3.6
source activate tensorflowGPU
printf '\nsource activate tensorflowGPU' >> ~/.bashrc
pip install --upgrade pip
pip install numpy matplotlib scikit-learn scipy seaborn PyQt5 pillow # for seaborn
printf 'y' | pip install numba
pip install umap-learn
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp36-cp36m-linux_x86_64.whl

# source deactivate
# printf 'y' | conda create -n tensorflowGPUSource python=3.6
# source activate tensorflowGPUSource
# printf '\nsource activate tensorflowGPUSource' >> ~/.bashrc

# pip install six numpy wheel matplotlib scikit-learn scipy seaborn PyQt5 pillow # for seaborn

# curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
# sudo apt-get update
# sudo apt-get install -y build-essential curl git m4 ruby texinfo libbz2-dev libcurl4-openssl-dev libexpat-dev libncurses-dev zlib1g-dev
# sudo apt-get install -y python-software-properties debconf-utils
# sudo add-apt-repository -y ppa:webupd8team/java
# sudo apt-get install -y python3-numpy python3-dev python3-pip python3-wheel

# sudo apt-get update
# echo debconf shared/accepted-oracle-license-v1-1 select true | sudo debconf-set-selections
# echo debconf shared/accepted-oracle-license-v1-1 seen true | sudo debconf-set-selections
# echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | sudo debconf-set-selections
# printf '\n' | sudo add-apt-repository ppa:webupd8team/java
# sudo apt-get update
# sudo apt-get install -y oracle-java8-installer
# sudo apt-get install -y oracle-java8-set-default

# echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# sudo apt-get update
# sudo apt-get install -y bazel 
# printf 'Y\n' |sudo apt-get upgrade bazel
# bazel version

# printf '\n' | ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install)"
# printf '\nexport PATH="/home/linuxbrew/.linuxbrew/bin:$PATH"' >> ~/.bashrc
# export PATH="/home/linuxbrew/.linuxbrew/bin:$PATH"
# brew install gcc
# brew install coreutils
# sudo apt-get update
# sudo apt-get install -y g++-4.8

# mkdir ~/tensorflowSource
# mkdir ~/tensorflowSource/tensorflow
# git clone https://github.com/tensorflow/tensorflow ~/tensorflowSource/tensorflow
# cd ~/tensorflowSource/tensorflow
# printf '\n\ny\nY\n\n\n\n\n\n\n\ny\n\n\n\n/usr/bin/gcc-4.8\n5.1.10\n\n\n\n' | ./configure
# bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k //tensorflow/tools/pip_package:build_pip_package
# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# cd /tmp/tensorflow_pkg/
# pip install `ls | head -n 1`
# cd ~







# bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k //tensorflow/tools/pip_package:build_pip_package
# bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 -k --local_resources 2048,.5,1.0 --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package

# sudo apt-get autoremove bazel
# sudo apt-get autoremove oracle-java8-installer

# alias activatetensorflow="source activate tensorflowSource"
# activatetensorflow
