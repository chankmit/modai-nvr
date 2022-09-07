FROM python:3.6

WORKDIR /app 
RUN apt-get update \
 && apt-get install -y sudo
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER docker
RUN sudo apt-get update -y && sudo apt-get upgrade -y
RUN sudo apt-get install git libatlas-base-dev gfortran libhdf5-serial-dev hdf5-tools python3-dev nano locate libfreetype6-dev python3-setuptools protobuf-compiler libprotobuf-dev openssl libssl-dev libcurl4-openssl-dev cython3 libxml2-dev libxslt1-dev -y

RUN cd /dev
RUN sudo apt-get -y install cmake protobuf-compiler

RUN sudo apt-get install build-essential pkg-config libtbb2 libtbb-dev libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libavresample-dev libtiff-dev libjpeg-dev libpng-dev python-tk libgtk-3-dev libcanberra-gtk-module libcanberra-gtk3-module libv4l-dev libdc1394-22-dev -y
RUN sudo wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
RUN sudo wget -O opencv_contrib.zip https://www.raoyunsoft.com/opencv/opencv_contrib/opencv_contrib-4.1.2.zip
RUN sudo unzip opencv.zip && sudo unzip opencv_contrib.zip
RUN sudo mv opencv-4.1.2 opencv && sudo mv opencv_contrib-4.1.2 opencv_contrib
RUN cd opencv
RUN sudo mkdir build
RUN cd build
RUN sudo cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D WITH_CUDA=ON \
-D CUDA_ARCH_PTX="" \
-D CUDA_ARCH_BIN="5.3,6.2,7.2" \
-D WITH_CUBLAS=ON \
-D WITH_LIBV4L=ON \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_opencv_java=OFF \
-D WITH_GSTREAMER=ON \
-D WITH_GTK=ON \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_EXAMPLES=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=/dev/opencv_contrib/modules .. 
RUN sudo make -j4 
RUN sudo make install

RUN cd ..
RUN cd ..
RUN sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev -y && sudo -H pip3 install future -y && sudo pip3 install -U --user wheel mock pillow -y && sudo -H pip3 install --upgrade setuptools Cython -y
RUN sudo -H pip3 install gdown 
RUN sudo gdown https://drive.google.com/uc?id=1TqC6_2cwqiYacjoLhLgrZoap6-sVL2sd
RUN sudo -H pip3 install torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl
RUN sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
RUN sudo pip3 install -U pillow
RUN sudo gdown https://drive.google.com/uc?id=1C7y6VSIBkmL2RQnVy8xF9cAnrrpJiJ-K
RUN sudo -H pip3 install torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl
RUN sudo git clone https://github.com/davisking/dlib.git
RUN cd dlib
RUN sudo mkdir build
RUN cd build
RUN cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
RUN cmake --build .
RUN cd ..
RUN sudo python3 setup.py install --set DLIB_USE_CUDA=1 
RUN sudo -H python3 setup.py install --set USE_AVX_INSTRUCTIONS=1 --set DLIB_USE_CUDA=1
RUN cd ..
RUN sudo -H python3 install face_recognition
RUN sudo -H python3 install streamlit

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit", "run"]

CMD ["app_face.py"]