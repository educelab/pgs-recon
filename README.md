# PGS Recon
A Python-based pipeline for reconstructing photogrammetry datasets. Includes a CMake project for compiling the 
required executables.

## Installation
### Requirements
The Python script uses executables provided by the OpenMVG and OpenMVS projects. The included CMake project will 
compile both of these projects as well as their dependencies. Before configuring the CMake project, please preinstall 
the following dependencies:
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

### Pipelines
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
In addition to VCG, OpenMVG, and OpenMVS, the CMake project also compiles a number of required software libraries. We 
provide corresponding CMake flags to control the compilation of these libraries. To use a system-provided version of 
these libraries, set the library's flag to `OFF`:

```cmake
BUILD_EIGEN: If ON, builds Eigen 3.2
BUILD_JPEG: If ON, builds libjpeg
BUILD_JPEG_TURBO: If ON, builds libjpeg-turbo (depends BUILD_JPEG=ON)
BUILD_OPENCV: If ON, builds OpenCV
BUILD_CGAL: If ON, builds CGAL
```

## Docker
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
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
  --build-arg USE_CUDA=ON \
  -t pgs-recon:dev-cuda \
  .
```

`BASE_IMAGE` should be an `nvidia/cuda:*-cudnn-devel-ubuntu*` Docker image. 
While this can theoretically be set to any Ubuntu and CUDA version, this has 
only been tested on:
 - CUDA 12.4, Ubuntu 22.04 
 - CUDA 12.8, Ubuntu 24.04

`USE_CUDA` should be either `ON` or `OFF [default]`. If `USE_CUDA=OFF`, CUDA 
will not be used even if you provide a CUDA-enabled base image.

## Apptainer
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