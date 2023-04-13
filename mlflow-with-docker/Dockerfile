FROM continuumio/miniconda3

# Create the environment using conda
RUN conda install -c anaconda jupyter
RUN conda install -c conda-forge mlflow
RUN conda install -c anaconda scikit-learn
RUN conda install -c anaconda psycopg2
RUN conda install -c anaconda boto3

# Set up SSH
RUN apt-get update && apt-get install -y openssh-server
RUN useradd -m -s /bin/bash dockeruser
RUN mkdir /var/run/sshd
RUN echo 'dockeruser:123' | chpasswd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]