FROM tensorflow/tensorflow:1.5.0-devel-gpu-py3

ENV NB_USER fru
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown fru /src && \
    chown fru /data

USER fru

WORKDIR /src

#CMD jupyter notebook --port=8888 --ip=0.0.0.0
CMD tensorboard --logdir /src/log --host 0.0.0.0
