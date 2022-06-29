FROM continuumio/miniconda3
RUN apt-get update
RUN apt-get install build-essential -y
RUN useradd -ms /bin/bash voice_smith
WORKDIR /home/voice_smith
COPY ./assets /home/voice_smith/assets
USER voice_smith
