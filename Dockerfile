# A dockerfile must always start by importing the base image.
# We use the keyword 'FROM' to do that.
# In our example, we want import the python image.
# So we write 'python' for the image name and 'latest' for the version.

### REBUILD PYMESH but with CUDA
# Import pymesh/pymesh so we can steal the whl from it later
FROM pymesh/pymesh:py3.5 as pymesh

# This is the base image for the final container 
FROM nvcr.io/nvidia/tensorflow:18.12-py3

## Copied from the Docker file at https://github.com/PyMesh/PyMesh/blob/main/docker/py3.5/Dockerfile
## Rebuild Pymesh by:
## 1) installing dependancies
## 2) building third_party apps 
## 3) copy whl from pymesh image and install

## 1) Install dependancies
WORKDIR /root/
ARG NUM_CORES=4

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y \
	gcc \
	g++ \
	git \
	curl \
	wget \
	vim \
	dssp \
	libgl1-mesa-glx \
	libgmp-dev \
	libmpfr-dev \
	libgmpxx4ldbl \
	libboost-dev \
	libboost-thread-dev \
	zip unzip patchelf && \
	apt-get clean && \
	git clone --single-branch -b main https://github.com/PyMesh/PyMesh.git

RUN apt remove -y cmake
RUN pip install --upgrade pip
RUN pip install --upgrade cmake

## 2) Build third_party apps
ENV PYMESH_PATH /root/PyMesh
ENV NUM_CORES $NUM_CORES
WORKDIR $PYMESH_PATH

RUN git submodule update --init && \
	pip install -r $PYMESH_PATH/python/requirements.txt && \
	./setup.py bdist_wheel && \
	rm -rf build_3.6 third_party/build

## 3) Copy whl from pymesh image and install
## Instead of running the patch_wheel.py script (which throws errors)
## Just copy the whl from the docker image where its already patched
COPY --from=pymesh /root/PyMesh/dist/pymesh2*.whl dist/

RUN pip install dist/pymesh2*.whl && \
	python -c "import pymesh; pymesh.test()"

#RUN git clone --single-branch https://github.com/LPDI-EPFL/masif

# DOWNLOAD/INSTALL APBS
RUN mkdir /install
WORKDIR /install
RUN git clone https://github.com/Electrostatics/apbs-pdb2pqr
WORKDIR /install/apbs-pdb2pqr
RUN ls
RUN git checkout b3bfeec
RUN git submodule init
RUN git submodule update
RUN ls
RUN cmake -DGET_MSMS=ON apbs
RUN make
RUN make install
RUN cp -r /install/apbs-pdb2pqr/apbs/externals/mesh_routines/msms/msms_i86_64Linux2_2.6.1 /root/msms/
RUN curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py
RUN python get-pip.py

# INSTALL PDB2PQR
WORKDIR /install/apbs-pdb2pqr/pdb2pqr
RUN git checkout b3bfeec
RUN apt-get install -y python2.7 
RUN python2.7 scons/scons.py install

# Setup environment variables 
ENV MSMS_BIN /usr/local/bin/msms
ENV APBS_BIN /usr/local/bin/apbs
ENV MULTIVALUE_BIN /usr/local/share/apbs/tools/bin/multivalue
ENV PDB2PQR_BIN /root/pdb2pqr/pdb2pqr.py

# DOWNLOAD reduce (for protonation)
WORKDIR /install
RUN git clone https://github.com/rlabduke/reduce.git
WORKDIR /install/reduce
RUN make install
RUN mkdir -p /install/reduce/build/reduce
WORKDIR /install/reduce/build/reduce
RUN cmake /install/reduce/reduce_src
WORKDIR /install/reduce/reduce_src
RUN make
RUN make install

# Install python libraries
RUN apt-get install -y libffi-dev
RUN pip3 install matplotlib 
RUN pip3 install ipython Biopython scikit-learn networkx open3d==0.8.0 dask packaging
#RUN pip install StrBioInfo 

# Clone masif
WORKDIR /

# We need to define the command to launch when we are going to run the image.
# We use the keyword 'CMD' to do that.
CMD [ "bash" ]
