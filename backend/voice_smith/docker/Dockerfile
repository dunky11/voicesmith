FROM nvcr.io/nvidia/nemo:22.01
# MFA shouldn't be run as root user
RUN useradd --create-home --shell /bin/bash voice_smith
RUN apt-get update
RUN pip install fire
WORKDIR /home/voice_smith
USER voice_smith
RUN conda create --name voice_smith
SHELL ["conda", "run", "--name", "voice_smith", "/bin/bash", "-c"]
RUN conda install -c conda-forge montreal-forced-aligner=2.0.0rc1=py39h82024d5_0
RUN pip install fire
RUN mfa model download g2p english_g2p
RUN mfa model download acoustic english
RUN conda init bash
COPY ./to_copy .
