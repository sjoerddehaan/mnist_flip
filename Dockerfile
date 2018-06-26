FROM jupyter/scipy-notebook:5811dcb711ba
MAINTAINER Sjoerd de Haan
USER jovyan
RUN conda install pytorch-cpu torchvision-cpu -c pytorch
# RUN pip install -r ~/requirements.txt
# RUN jupyter contrib nbextension install --sys-prefix
# RUN jupyter nbextension enable --py widgetsnbextension
# RUN jupyter nbextension enable python-markdown/main
# RUN jupyter nbextension enable codefolding/main 
# RUN jupyter nbextension enable hinterland/hinterland
# RUN jupyter nbextension enable python-markdown/main
# RUN jupyter nbextension enable toc2/main
# RUN jupyter nbextension enable spellchecker/main
# RUN jupyter nbextension enable collapsible_headings/main
run git clone https://github.com/sjoerddehaan/mnist_flip
