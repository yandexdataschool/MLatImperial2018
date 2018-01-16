FROM andrewosh/binder-base
MAINTAINER Alexander Panin <justheuristic@gmail.com>
USER root

RUN apt-get -qq update

RUN apt-get install -y gcc-4.9 g++-4.9 libstdc++6 wget unzip
RUN apt-get install -y libopenblas-dev liblapack-dev libsdl2-dev libboost-all-dev 
RUN apt-get install -y cmake zlib1g-dev libjpeg-dev 
RUN apt-get install -y xvfb libav-tools xorg-dev python-opengl python3-opengl
RUN apt-get -y install swig3.0
RUN ln -s /usr/bin/swig3.0 /usr/bin/swig


USER main
RUN pip install --upgrade pip
RUN pip install --upgrade --ignore-installed setuptools  #fix https://github.com/tensorflow/tensorflow/issues/622
RUN pip install --upgrade sklearn tqdm nltk editdistance joblib catboost xgboost
RUN pip install --upgrade gym[all]
RUN pip install --upgrade http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 
RUN pip install --upgrade torchvision 



RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade pip

# python3: fix `GLIBCXX_3.4.20' not found - conda's libgcc blocked system's gcc-4.9 and libstdc++6
RUN bash -c "conda update -y conda && source activate python3 && conda uninstall -y libgcc && source deactivate"
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade matplotlib numpy scipy pandas

RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade sklearn tqdm nltk editdistance joblib catboost xgboost
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade --ignore-installed setuptools #fix broken setuptools
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade gym[all]
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade torchvision

