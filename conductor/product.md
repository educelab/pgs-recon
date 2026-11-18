# Product Definition

## Name

PGS Recon (package and CLI prefix: `pgs-recon`)

## Description

A Python pipeline that reconstructs textured 3D models from photogrammetry image
sets by orchestrating OpenMVG (SfM) and OpenMVS (MVS).

## Problem

Running OpenMVG/OpenMVS by hand is a long, error-prone chain of binaries and
flags. There is no reproducible, resumable, recorded way to go from captures to a
textured mesh.

## Users

- **Primary:** EduceLab researchers processing PGS scans and other capture sets,
  via Docker/Apptainer on workstations and the cluster.
- **Secondary:** the wider cultural-heritage imaging community.

## Key Goals

1. **Reproducible and resumable.** Every run is recorded in a manifest
   (`pgs-recon.json`) and can be resumed or re-planned without silently
   inheriting stale state.
2. **Easy deployment.** Published Docker/Apptainer images run identically on
   workstations and the cluster.

## Scope

- `pgs-recon`: the full SfM → MVS → textured mesh pipeline.
- Standalone `pgs-*` utilities (Python apps and C++ tools): import, conversion,
  masking, retexturing, decimation, localization, quality checks.
