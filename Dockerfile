FROM ubuntu:22.04
LABEL org.opencontainers.image.authors="Seth Parker <c.seth.parker@uky.edu>"
LABEL org.opencontainers.image.title="PGS Recon"
LABEL org.opencontainers.image.description="A photogrammetry reconstruction pipeline"
LABEL org.opencontainers.image.source="https://gitlab.com/educelab/pgs-recon"

# Set environment variables
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Install apt dependencies and build photogrammetry dependencies
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
      build-essential \
      bzip2 \
      cmake \
      curl \
      doxygen \
      git \
      graphviz \
      imagemagick \
      libboost-iostreams-dev \
      libboost-program-options-dev \
      libboost-serialization-dev \
      libboost-system-dev \
      libboost-thread-dev \
      libbz2-dev \
      libgmp3-dev \
      libjpeg-turbo8-dev \
      libqt5opengl5-dev \
      libmpfr-dev \
      libqt5svg5-dev \
      libqt5x11extras5-dev \
      locales \
      nano \
      ninja-build \
      python3-dev \
      python3-pip \
      python3-venv \
      qtbase5-dev \
      qt5-qmake \
      software-properties-common \
      tzdata \
      wget \
      xz-utils \
      zlib1g-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen  \
    && locale-gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && apt clean && apt autoremove -y --purge && rm -rf /var/lib/apt/lists/*

# Install PGS Recon dependencies
COPY . /usr/local/educelab/pgs-recon
RUN cmake \
      -S /usr/local/educelab/pgs-recon/dependencies/ \
      -B /usr/local/educelab/build/ \
      -DCMAKE_INSTALL_PREFIX=/usr/local/ \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_JPEG=OFF \
      -GNinja  \
    && cmake --build /usr/local/educelab/build/ \
    && rm -rf /usr/local/educelab/build

# Install ExifTool
RUN mkdir -p /usr/local/educelab/exiftool/  \
    && cd /usr/local/educelab/exiftool/ \
    && export EXIFTOOL_VER=$(curl --silent https://exiftool.org/ver.txt) \
    && curl -O -L https://exiftool.org/Image-ExifTool-${EXIFTOOL_VER}.tar.gz \
    && tar -xzf Image-ExifTool-${EXIFTOOL_VER}.tar.gz && cd Image-ExifTool-${EXIFTOOL_VER}/ \
    && perl Makefile.PL && make test && make install

# Make venv and cleanup
RUN python3 -m venv /usr/local/educelab/pgs-recon/.venv \
    && . /usr/local/educelab/pgs-recon/.venv/bin/activate \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && python3 -m pip install --editable /usr/local/educelab/pgs-recon \
    && chmod --recursive a+rw /usr/local/educelab/pgs-recon/ \
    && chmod a+rw /usr/local/lib/openMVG/sensor_width_camera_database.txt \
    && git config --global credential.helper "cache --timeout=3600" \
    && git config --global --add safe.directory /usr/local/educelab/pgs-recon/ 

ENV PATH="/usr/local/educelab/pgs-recon/.venv/bin:$PATH"
CMD ["pgs-recon", "--help"]
