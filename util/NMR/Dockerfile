FROM chainer/chainer:v4.4.0-python2
COPY 32770_pred.png .
RUN apt-get update \
  && apt-get install -y git
RUN git clone https://github.com/hiroharu-kato/neural_renderer.git
RUN git clone https://github.com/hiroharu-kato/mesh_reconstruction.git
COPY models mesh_reconstruction/data/models
COPY eval_images mesh_reconstruction/data/eval_images

RUN apt-get install imagemagick -y
RUN apt-get install nano

RUN pip install scipy==0.19.0 scikit-image==0.10 tqdm==3.7.1 Pillow==4.3.0