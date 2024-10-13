FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# install packages
RUN apt-get update && \
	apt-get install -y python3 python3-pip bash curl && \
	apt-get install -y neovim && \
	apt-get install -y git && \
	apt-get install -y cargo

# setup
WORKDIR /root/models_testing/
COPY . .

# setup pyenergibridge
RUN cd; git clone https://github.com/tdurieux/EnergiBridge.git && \
	cd EnergiBridge; cargo build -r && \
	cd; git clone https://github.com/luiscruz/pyEnergiBridge.git && \
	cd pyEnergiBridge; pip install . && \
	echo '{"binary_path": "/root/EnergiBridge/target/release/energibridge"}' > /root/models_testing/pyenergibridge_config.json && \
	echo '{"binary_path": "/root/EnergiBridge"}' > /root/.pyenergibridge_config.json

# install python repos
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# start an interactive bash shell
CMD ["/bin/bash"]