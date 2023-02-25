FROM chainer/chainer:v4.4.0-python2
RUN apt-get update \
  && apt-get install -y git
RUN git clone https://github.com/Kerstin08/neural_renderer
RUN git clone https://github.com/Kerstin08/mesh_reconstruction
COPY models mesh_reconstruction/data/models
COPY eval_images mesh_reconstruction/data/eval_images

RUN apt-get install imagemagick -y

RUN pip install scipy==0.19.0 scikit-image==0.10 tqdm==3.7.1 Pillow==4.3.0