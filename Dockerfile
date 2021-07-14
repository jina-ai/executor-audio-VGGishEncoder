FROM jinaai/jina:2.0.7-py38-perf

# install git
RUN apt-get -y update && apt-get install -y git curl libsndfile1

RUN pip install tensorflow_cpu==2.2.0

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace

COPY scripts/download_model.sh /download_model.sh
RUN /download_model.sh

# setup the workspace
# COPY ./ /workspace

# RUN ./scripts/download_model.sh

# ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
ENTRYPOINT [ "/bin/bash" ]
