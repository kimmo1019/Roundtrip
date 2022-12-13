FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install python2.7
RUN apt-get -y install python-pip
RUN pip install pip --upgrade
RUN apt-get -y install wget
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888
USER root
## this is to make jupyter notbook work with dokcer containers, you can comment this if don't want to use jupyter
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
