FROM jupyter/scipy-notebook:5811dcb711ba
MAINTAINER Sjoerd de Haan
USER jovyan
RUN conda install -y pytorch-cpu=0.4.0 \
   torchvision-cpu=0.2.1 \
   -c pytorch
RUN pip install jupyter_contrib_nbextensions && \
  jupyter nbextension enable toc2/main && \
  jupyter nbextension enable spellchecker/main
run git clone https://github.com/sjoerddehaan/mnist_flip
