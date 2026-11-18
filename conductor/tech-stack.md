# Tech Stack

## Languages

| Language | Version | Where |
| -------- | ------- | ----- |
| Python | **target ≥ 3.12** (currently `>=3.9` in `setup.cfg`; 3.9 is EOL) | `pgs_recon/` |
| C++ | C++17; C++20 for `pgs-localize` only (ADR 0011) | `dependencies/utilities/` |
| CMake | ≥ 3.17 (superbuild), ≥ 3.16 (utilities) | `dependencies/` |

No frontend, backend framework or database. All interfaces are CLIs.

## Python

- Packaging: setuptools (`setup.cfg`, `pyproject.toml`); version lives in `setup.cfg`.
- CLI: `configargparse`.
- Core deps (`requirements.txt`): `numpy>=2`, `scipy`, `scikit-image`,
  `opencv-python-headless>=4.10`, `imageio`, `tifffile`, `imagecodecs`, `vtk`,
  `matplotlib`, `tqdm`, `PySfMUtils` (`sfm_utils`), `educelab-imgproc`,
  `PyExifTool` (needs the `exiftool` binary).
- Tests: stdlib `unittest` (`python3 -m unittest discover -s tests`).

## C++ / External Toolchain

Built by the CMake superbuild in `dependencies/`, installed under a prefix
(`/usr/local` in images):

- OpenMVG (SfM), OpenMVS v2.4 (MVS, with local patches in `dependencies/patches/`)
- VCG, CGAL, Ceres, Eigen, OpenCV, libjpeg(-turbo), nanoflann, Boost
- In-tree utilities: `pgs-global-scaler`, `pgs-sfm-orient`,
  `pgs-generate-markers`, `pgs-decimate`, `pgs-localize`
  (bvh v2 and libcore via `FetchContent`)

## Infrastructure

### Current

- **Docker** images on `ghcr.io/educelab/pgs-recon` (`edge` + semver tags),
  amd64 and arm64, CPU and CUDA 12.4 / 12.8 variants, all on **Ubuntu 22.04**
  bases. Built by GitHub Actions (`.github/workflows/build_docker.yml`).
- **Apptainer** (`apptainer/pgs-recon.def`) with SLURM submit scripts for the
  cluster.
- **CI:** GitLab CI (`.gitlab-ci.yml`) runs the test suite bare, with Python
  deps, and in the published image (`test:in-image`), then builds dependencies
  on Ubuntu 22.04 / 24.04. GitHub Actions also has `ci.yml` and `deps-build.yml`.

### Planned: Ubuntu 24.04 migration

Raising the Python minimum to 3.12 means:

- Docker/Apptainer bases → `ubuntu:24.04` and the CUDA `*-ubuntu24.04` variants.
- GitHub Actions runners → `ubuntu-24.04` / `ubuntu-24.04-arm`; GitLab jobs pinned
  to 22.04 move to 24.04; `ci/apt-ubuntu-jammy` gets a noble equivalent.
- `setup.cfg` `python_requires = >=3.12`.
- Drop Ubuntu 22.04 from the build matrix.
