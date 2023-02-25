FROM pymesh/pymesh

ENV LD_LIBRARY_PATH=/usr/local/lib/

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libglu1-mesa
RUN apt-get install freeglut3
RUN apt-get -y install google-perftools
RUN apt-get install libtbb-dev -y
RUN ln -s /usr/lib/x86_64-linux-gnu/libtbb.so /usr/lib/x86_64-linux-gnu/libtbb_preview.so.2
RUN apt-get install libjpeg-dev -y
RUN ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so.62  /usr/lib/x86_64-linux-gnu/libjpeg.so.8
RUN wget -q -O /tmp/libpng12.deb http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb \
  && dpkg -i /tmp/libpng12.deb \
  && rm /tmp/libpng12.deb

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir h5py joblib trimesh==3.14.1 mkl==2019.0 Pillow

CMD python -u /workspace/preprocessing/create_point_sdf_grid.py --thread_num 1 --category 'car'