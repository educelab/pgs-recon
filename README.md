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

## Utilities
In addition to the main `pgs-recon` pipeline, this project ships several
standalone tools. All of them can be launched through the Docker image in the
same way as `pgs-recon` (e.g. `docker run ... pgs-sfm-orient --help`).

### `pgs-sfm-orient`
Centers, orients, and (optionally) scales a reconstructed mesh using the
EduceLab sample square / ArUco markers detected directly in the **SfM scene
images**. It is the SfM-based counterpart to `pgs-center`: where `pgs-center`
detects the sample square in the mesh's UV texture (which requires a coherent,
reordered texture map), `pgs-sfm-orient` detects and triangulates the markers
from the original images, so it works regardless of how the mesh was textured.

The translation comes from the mesh's oriented-bounding-box center (so the
object lands at the origin), the orientation and scale come from the markers,
and the result is written as a transformed mesh and/or a 4×4 similarity
transform. The input mesh must already be in the SfM coordinate frame.

```shell
docker run -v .:/working ghcr.io/educelab/pgs-recon \
  pgs-sfm-orient \
    -i /working/recon/sfm/sfm_data.bin \
    --input-mesh /working/recon/mvs/my-object.obj \
    -o /working/recon/mvs/my-object-centered.obj \
    --save-transform /working/recon/orient.npy \
    -s 0.47
```

Key options:
* `-i, --input-scene` — the SfM scene file (markers are detected in its images).
* `--input-mesh` — mesh (`.obj`/`.ply`) in the SfM frame; enables OBB-center
  translation and the bounding-box orientation fallback.
* `-o, --output-mesh` — write the transformed mesh (requires `--input-mesh`).
* `--save-transform` — write the 4×4 transform as a NumPy `.npy`, compatible
  with `pgs-center --load-transform` and `pgs-calibrate`/`pgs-retexture
  --sfm-transform`.
* `-s, --marker-size` — marker size in the desired world units (required unless
  `--no-scale` or `--orient-method bbox`).
* `--orient-method {auto,aruco,bbox}` — orientation source (default `auto`: use
  markers if detected, otherwise fall back to the mesh bounding box). `aruco`
  fails if no markers are found; `bbox` ignores markers and requires a mesh.
* `--no-scale` — skip scale estimation (output rotation + translation only).

At least one of `--output-mesh` or `--save-transform` is required.

### Camera calibration file format
`pgs-calibrate` reads and writes camera parameters in a single plain-text
**camera calibration file**. It is a flat list of `key value` entries, one per
line; blank lines and lines beginning with `#` are ignored, and unrecognized
keys are skipped (so the same file can carry both an intrinsic and a pose, and
each consumer reads only what it needs).

| Key | Meaning |
| --- | --- |
| `fx`, `fy` | Focal length in **pixels** (x and y). `fy` defaults to `fx` if omitted. OpenMVG uses a single focal, so the two should match. |
| `cx`, `cy` | Principal point in **pixels**. |
| `width`, `height` | Image resolution (pixels) the intrinsic is calibrated at. |
| `k1`, `k2`, `k3` | Radial distortion coefficients (OpenCV/OpenMVG order). Optional; absent means no distortion. |
| `pose` | 16 whitespace-separated floats: a **row-major 4×4 world-to-camera** matrix in OpenCV convention (`x_cam = R·X + t`). |

Example (an overhead camera with mild barrel distortion):

```
# my overhead RGB camera
fx 18250.0
fy 18250.0
cx 3000.0
cy 2000.0
width 6000
height 4000
k1 -0.082
k2 0.011
k3 0.0
pose 0.9998 0.0011 -0.0203 12.4 -0.0009 0.9999 0.0102 -8.1 0.0203 -0.0102 0.9997 423.7 0 0 0 1
```

Two flags use this format:

* **`pgs-calibrate --intrinsic <file>`** reads it as a *precalibrated* query
  intrinsic. It requires the intrinsic keys (`fx`, `cx`, `cy`, `width`,
  `height`); `fy` and the `k*` distortion are optional, and any `pose` is
  ignored (the pose is what calibration solves for). The intrinsic is scaled to
  the query image's resolution automatically, and the distortion is honored —
  OpenMVG undistorts the query before resectioning. This is the stable,
  recommended path for long-focal overhead cameras with few feature matches.
  (For a focal-only calibration you can instead pass `--focal-length` in pixels,
  or `--focal-length-mm` together with `--pixel-size` (mm/px) or `--sensor-width`
  (mm); both assume a centered principal point and no distortion.)
* **`pgs-calibrate --save-camera-file <file>`** writes the solved calibration in
  this format (intrinsic + `pose`, with `k*` emitted only when non-zero). It is
  consumed by [registration-toolkit](https://github.com/educelab/registration-toolkit)
  and can be fed straight back into `--intrinsic`.

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
 - CUDA 12.8, Ubuntu 22.04 (with and without CUDNN)

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