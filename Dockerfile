FROM --platform=linux/amd64 debian:trixie-slim

SHELL ["/bin/bash", "--login", "-c"]

# install apt dependencies
RUN apt-get update
RUN apt-get -y install curl

# install condaforge
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" &&\
    bash Miniforge3-Linux-x86_64.sh -b -p /condaforge &&\
    /condaforge/bin/conda init bash &&\
    rm "Miniforge3-Linux-x86_64.sh"


# create environment
COPY docker_environment.yaml /DBSegment/docker_environment.yaml
RUN conda env create -f /DBSegment/docker_environment.yaml --quiet &&\
    conda clean -afy

# useful for debugging, avoids reinstallation of all dependencies
# RUN conda run -n dbsegment pip install nibabel==5.2.1 scipy==1.13.1 torch==2.3.0 requests==2.32.2  antspyx==0.4.2 nnunet==1.7.1

COPY .. /DBSegment
RUN cd /DBSegment && conda run -n dbsegment pip install . --no-cache-dir &&\
    conda run -n dbsegment pip cache purge

ENTRYPOINT [ "bin/bash", "--login", "-c", "/condaforge/bin/conda run -n dbsegment DBSegment \"$@\" -i /input -o /output -mp /models" ]

