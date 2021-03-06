FROM jupyter/scipy-notebook:5811dcb711ba
MAINTAINER Sjoerd de Haan
USER jovyan
RUN conda install -y pytorch-cpu=0.4.0 \
   torchvision-cpu=0.2.1 \
   -c pytorch
USER root
 RUN pip install jupyter_contrib_nbextensions && \
  jupyter contrib nbextension install && \
  jupyter nbextension enable spellchecker/main
USER jovyan
RUN conda install -y tqdm=4.23.4
run git clone https://github.com/sjoerddehaan/mnist_flip
