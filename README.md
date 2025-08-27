# PGS Recon
A Python-based pipeline for reconstructing photogrammetry datasets using 
[OpenMVG](https://github.com/openMVG/openMVG) and 
[OpenMVS](https://github.com/cdcseacave/openMVS).

## Usage
The simplest way to get started is to pull the Docker image of this project and 
run `pgs-recon` on your directory of images:

```shell
# Download the image
docker pull ghcr.io/educelab/pgs-recon:latest

# Run reconstruction on a directory of images in the current working directory
# Flags:
#  -v .:/working       - Mounts the current working directory to '/working' 
#                        inside the container
#  -i /working/images  - Path to the images inside the container
#  -o /working/recon   - Output directory inside the container
#  --name my-object    - Descriptive name for the scanned object/scene. This is 
#                        used to name the output file. If not provided, defaults
#                        to a name derived from the current timestamp and the 
#                        name of the input directory
docker run -v .:/working ghcr.io/educelab/pgs-recon:latest \
  pgs-recon -i /working/images/ -o /working/recon/ --name my-object
```

Upon successful completion of the pipeline, your reconstructed model can be 
found in `recon/mvs/my-object.obj`.

### Docker images
We provide multi-architecture (x86, arm64) Docker images in the 
[GitHub Container Registry](https://github.com/educelab/pgs-recon/pkgs/container/pgs-recon).
Simply pull our container and Docker will select the appropriate image for your
host platform:
```shell
# Pull the latest release
docker pull ghcr.io/educelab/pgs-recon:latest

# Pull the latest edge version
docker pull ghcr.io/educelab/pgs-recon:edge

# Pull a specific version
docker pull ghcr.io/educelab/pgs-recon:1.7.1
```

CUDA-enabled images are available by appending `-cudaX.X` to any of the standard
tags. We currently only provide images for CUDA 12.4 and 12.8:
```shell
# Pull the latest CUDA 12.4 release
docker pull ghcr.io/educelab/pgs-recon:latest-cuda12.4

# Pull the latest CUDA 12.8 release
docker pull ghcr.io/educelab/pgs-recon:latest-cuda12.8
```

All project tools can be launched directly using `docker run`:
```shell
$ docker run ghcr.io/educelab/pgs-recon pgs-recon --help
usage: pgs-recon [-h] [--config CONFIG] --input INPUT --output OUTPUT
                 [--name NAME] [--file-type {ply,obj}] [--focal-length n]
                 [--new-importer | --no-new-importer]
                 [--import-pgs-scan | --no-import-pgs-scan | -p]
                 [--import-calib IMPORT_CALIB]
...
```

## Install from source
### Install dependencies
The Python scripts use executables provided by the OpenMVG and OpenMVS projects. 
The included CMake project will compile both of these projects and their 
dependencies. Before configuring the CMake project, please preinstall the 
following dependencies:
* CMake 3.17+
* Boost 1.70+
* GMP and MPFR
* ExifTool
* (Optional) NASM (Required by jpeg-turbo)
* (Optional) Ceres Solver
* (Optional) CUDA Toolkit

After the dependencies have been installed, configure and build the CMake 
project to compile the required executables:
```shell
cmake -S dependencies -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/
```

**Notes**:
* OpenMVS and OpenCV must be linked against the same version of libjpeg.

### Install the Python pipeline scripts
Use a recent version of `pip` to install the Python scripts:
```shell
# Requires Python 3.9+
python3 -m pip install .
```

After installation, the reconstruction script can be run from the shell:
```shell
pgs-recon --help
```

### Advanced Installation
#### Installation Location
By default, executables created by this CMake project will be installed to 
`dependencies/installed/`. The installation location can be changed by setting 
the CMake installation prefix flag:
```shell
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ ..
```

#### Disable compilation of extra libraries
In addition to VCG, OpenMVG, and OpenMVS, the CMake project also compiles a 
number of required software libraries. We provide corresponding CMake flags to 
control the compilation of these libraries. To use a system-provided version of 
these libraries, set the library's flag to `OFF`:

```cmake
BUILD_EIGEN: If ON, builds Eigen 3.2
BUILD_JPEG: If ON, builds libjpeg
BUILD_JPEG_TURBO: If ON, builds libjpeg-turbo (depends BUILD_JPEG=ON)
BUILD_OPENCV: If ON, builds OpenCV
BUILD_CGAL: If ON, builds CGAL
```

## Building a Docker image
Docker images can be built by running the following from the root of the 
project directory:
```shell
docker build -t pgs-recon:dev .
```

By default, this image only supports a CPU-based reconstruction pipeline. 
A CUDA-enabled image can be built by passing the `BASE_IMAGE` and `USE_CUDA`
build args:
```shell
docker build -t pgs-recon:dev \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --build-arg USE_CUDA=ON \
  -t pgs-recon:dev-cuda \
  .
```

`BASE_IMAGE` should be an `nvidia/cuda:*-devel-ubuntu*` Docker image 
[[link]](https://hub.docker.com/r/nvidia/cuda/tags?name=devel-ubuntu). 
While this can theoretically be set to any Ubuntu and CUDA version, this has 
only been tested on:
 - CUDA 12.4, Ubuntu 22.04 (with and without CUDNN)
 - CUDA 12.8, Ubuntu 24.04 (with and without CUDNN)

`USE_CUDA` should be either `ON` or `OFF [default]`. If `USE_CUDA=OFF`, CUDA 
will not be used even if you provide a CUDA-enabled base image.

## Building an Apptainer image
Apptainer images can be built by running the following from the root of the 
project directory:
```shell
apptainer build pgs-recon.sif apptainer/pgs-recon.def
```

By default, this image only supports a CPU-based reconstruction pipeline. 
A CUDA-enabled image can be built by passing the provided build args file for
your required CUDA version:
```shell
apptainer build pgs-recon.sif \
  --build-arg-file apptainer/buildargs-cuda12.4.env \
  apptainer/pgs-recon.def
```