FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz openmpi-bin libgl1-mesa-glx && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

USER keras

RUN conda install -y python=3.6 && \
    pip install --upgrade pip

RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl

RUN conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py && \
    pip install sklearn_pandas

RUN pip install keras==2.1.2 && \
    conda clean -yt

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /src

#CMD jupyter notebook --port=8888 --ip=0.0.0.0
CMD tensorboard --logdir /src/log --host 0.0.0.0
