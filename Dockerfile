# First Stage
FROM python:3.13

# add new user 'teli' and change to it
RUN useradd -ms /bin/bash teli
USER teli
WORKDIR /home/teli 

# install Python packages
RUN mkdir Downloads
WORKDIR /home/teli/Downloads
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
WORKDIR /home/teli

# Change work directory
RUN mkdir workspace
WORKDIR /home/teli/workspace

# CMD command
CMD ["/bin/bash"]
