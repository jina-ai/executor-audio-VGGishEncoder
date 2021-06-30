FROM jinaai/jina:2.0

# install git
RUN apt-get -y update && apt-get install -y git curl

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY ./ /workspace
WORKDIR /workspace

RUN ./scripts/download_model.sh

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
