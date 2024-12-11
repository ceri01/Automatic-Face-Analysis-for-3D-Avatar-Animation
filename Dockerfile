FROM python:3.9
LABEL authors="Daniele Ceribelli"
WORKDIR /app

RUN git clone https://github.com/OpenKinect/libfreenect2.git && \
    cd libfreenect2 || exit && \
    mkdir build && \
    cd build || exit && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2 && \
    make && \
    make install && \
    export LD_LIBRARY_PATH=$HOME/freenect2/lib && \
    cd ../.. && \
    export PATH=$PWD/libfreenect2:$PATH && \
    pip install -r requirements.txt \


ENTRYPOINT ["top", "-b"]