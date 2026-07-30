#!/bin/bash
#
# Example: run one pgs-recon reconstruction as a chain of Slurm jobs, each sized
# for the stages it runs.
#
#   job 1  import..convert          all OpenMVG stages     CPU, many cores
#   job 2  densify                  DensifyPointCloud      GPU
#   job 3  reconstruct..texture     mesh + refine + texture  CPU, high memory
#   job 4  post-process + deliver   reorder/center/copy out  CPU, small
#
# The point of the split is memory: RefineMesh needs far more than anything else
# in the pipeline, so a single whole-pipeline job has to be sized for its worst
# case -- wasting a large allocation on hours of cheap SfM, or getting OOM-killed
# after that work is already done. Here only job 3 asks for the big node.
#
# Each job runs `pgs-recon` against the same --output directory and picks up
# where the last stopped; state lives in that directory's metadata.json. Only
# job 1 needs -i/--name/-c: every argument it ran with is recorded and reloaded
# by the later jobs.
#
# RUN THIS FROM A LOGIN NODE. It submits the chain and exits -- do not `sbatch`
# it (its own sbatch calls would then run on a compute node).

set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: submit_recon_pipeline.sh <dtn_path_to_input_folder> <pgs-recon args>"
    echo "  Path should be a path accessible from dtn."
    echo "  Example:"
    echo "    ./submit_recon_pipeline.sh /mnt/gemini1-4/seales_uksr/herculaneum/Dailies/Preinterval/20211207_103351_7Dec2021_PHercPHerc1186Cr01_90d8c473"
    echo "  Extra arguments are passed to pgs-recon in job 1 (and inherited by the rest):"
    echo "    ./submit_recon_pipeline.sh <path> --describer-preset HIGH"
    echo "  Convert options and partitions come from the environment:"
    echo "    PGS_EXPOSURE=1.5 PGS_PART_HIMEM=CAC48M768_L ./submit_recon_pipeline.sh <path>"
    exit 2
fi

# --- Cluster resources -------------------------------------------------------
# Partition names below are LCC examples; override them for your cluster. Each
# job asks only for what its own stages need.
account="${PGS_ACCOUNT:-col_seales_uksr}"
part_cpu="${PGS_PART_CPU:-CAL48M192_L}"           # SfM: cores, ordinary memory
part_gpu="${PGS_PART_GPU:-V4V32_CAL48M192_L}"     # densify: needs a GPU
part_himem="${PGS_PART_HIMEM:-CAC48M768_L}"       # refine: needs the memory
cpus_cpu="${PGS_CPUS_CPU:-32}"
cpus_gpu="${PGS_CPUS_GPU:-16}"
cpus_himem="${PGS_CPUS_HIMEM:-32}"
mem_himem="${PGS_MEM_HIMEM:-512G}"
gpus="${PGS_GPUS:-1}"

# --- Important paths ---------------------------------------------------------
containers_dir="${PROJECT}/seales_uksr/containers"
pgs_recon_container="${containers_dir}/pgs-recon.sif"
# Densify runs on a GPU, so job 2 uses a CUDA build of the same container
# (apptainer/pgs-recon.def with USE_CUDA=ON, see buildargs-cuda12.4.env).
pgs_recon_cuda_container="${containers_dir}/pgs-recon-cuda.sif"
registration_toolkit_container="${containers_dir}/registration-toolkit.sif"
root_processing_dir="${PSCRATCH}/seales_uksr/herculaneum-processing"
recon_config_file="${PGS_CONFIG:-"${PSCRATCH}/seales_uksr/dri-experiments-drive/2021-pgs-recon/configs/pgs-recon-import-global.txt"}"
processed_dir="/mnt/gemini1-4/seales_uksr/herculaneum/Processed"

# Convert options
exposure="${PGS_EXPOSURE:-2.5}"
shadows="${PGS_SHADOWS:-50}"

# Generated options
dtn_path="$1"
job_name="$( basename "${dtn_path}" )"
# Everything after the input path is passed through to job 1. It is spliced into
# a generated heredoc, so quote it here or `--name "my object"` word-splits on
# the compute node. Guard the empty case: `printf '%q '` with no arguments still
# emits `''`, which pgs-recon rejects as an unrecognized argument.
extra_args=""
if (( $# > 1 )); then
  extra_args="$( printf '%q ' "${@:2}" )"
fi
processed_job_name="${job_name}_processed_$(date +'%Y%m%d_%H%M%S')"
processed_job_dir="${root_processing_dir}/${processed_job_name}"
submit_dir="$( pwd -P )"

# --- Preflight ---------------------------------------------------------------
# Fail on the login node rather than three jobs deep.
if ! ssh dtn exit ; then
  echo "ssh Host 'dtn' not found or connection unsuccessful. Please add Host 'dtn' to your ssh config"
  exit 2
fi

for overlay in "pgs-recon.overlay" "registration-toolkit.overlay"
do
  if ! test -f "${overlay}" ; then
    echo "Creating ${overlay}"
    dd if=/dev/zero of="${overlay}" bs=1M count=500 && mkfs.ext3 -F "${overlay}"
  fi
done

# Shared #SBATCH settings. --export makes rclone available inside the container.
# shellcheck disable=SC2054  # the commas belong inside --export=, not between elements
sbatch_common=(
  -A "${account}"
  --mail-type=FAIL
  --chdir "${submit_dir}"
  --export=ALL,SINGULARITY_BIND='/share/singularity/bin',SINGULARITYENV_PREPEND_PATH='/share/singularity/bin'
)

# `\${...}` below defers expansion to the compute node; `${...}` is expanded now.

# --- Job 1: stage in, convert to JPG, run every OpenMVG stage ----------------
# --to convert stops after openMVG2openMVS, so this job produces the interface
# scene.mvs that densify consumes and nothing more.
#
# --mvs-densify belongs HERE, on the run that establishes the pipeline shape:
# densify renames the whole mesh chain (scene_mesh.ply -> scene_dense_mesh.ply),
# so adding it on a later job would invalidate stages that job is not sized to
# rebuild. pgs-recon would notice and warn rather than silently skip them, but
# the chain would still need a re-run to catch up. (--mvs-refine is on by
# default, so refine is already in the shape.)
job1=$(sbatch --parsable "${sbatch_common[@]}" \
  --job-name="pgs-mvg-${job_name}" \
  --output="pgs-recon_mvg_%j_out.txt" \
  -p "${part_cpu}" --cpus-per-task="${cpus_cpu}" --mem=0 --time=1-00:00:00 \
  <<EOF
#!/bin/bash
set -e
module load ccs/singularity

echo "Copying dataset to lcc"
time rsync -a dtn:"${dtn_path%/}" "${root_processing_dir}/"

echo "Converting raw images to jpg"
time singularity run --overlay pgs-recon.overlay "${pgs_recon_container}" \\
  pgs-convert -i "${root_processing_dir}/${job_name}" \\
  --exposure="${exposure}" --shadows="${shadows}" \\
  -o "${processed_job_dir}/jpg"

echo "Running OpenMVG stages (import..convert)"
time singularity run --overlay pgs-recon.overlay "${pgs_recon_container}" \\
  pgs-recon -c "${recon_config_file}" \\
  -i "${processed_job_dir}/jpg" \\
  -o "${processed_job_dir}" \\
  -n "${job_name}" \\
  --to convert \\
  --mvs-densify \\
  --threads "\${SLURM_CPUS_PER_TASK}" \\
  ${extra_args}
EOF
)
printf '  job 1  %-22s %s  %s\n' "import..convert" "${part_cpu}" "${job1}"

# --- Job 2: densify on a GPU -------------------------------------------------
# No -i, no --name, no -c: job 1's arguments are read back from the manifest.
# --threads is a per-node setting, so it may differ freely between jobs.
job2=$(sbatch --parsable "${sbatch_common[@]}" \
  --job-name="pgs-densify-${job_name}" \
  --output="pgs-recon_densify_%j_out.txt" \
  --dependency="afterok:${job1}" \
  -p "${part_gpu}" --cpus-per-task="${cpus_gpu}" --gres=gpu:"${gpus}" \
  --time=12:00:00 \
  <<EOF
#!/bin/bash
set -e
module load ccs/singularity

echo "Densifying point cloud"
time singularity run --nv --overlay pgs-recon.overlay "${pgs_recon_cuda_container}" \\
  pgs-recon -o "${processed_job_dir}" \\
  --from densify --to densify \\
  --threads "\${SLURM_CPUS_PER_TASK}"
EOF
)
printf '  job 2  %-22s %s  %s\n' "densify" "${part_gpu}" "${job2}"

# --- Job 3: mesh, refine, texture on a high-memory node ----------------------
# --from reconstruct with no --to runs reconstruct, refine and texture. This is
# the only job that needs the big allocation.
#
# If it is OOM-killed, resubmit it unchanged and it resumes: whatever finished is
# recorded complete and skipped. To retry refine less aggressively instead, add
# an argument it owns and the stages downstream of it re-run too:
#   pgs-recon -o <dir> --from refine --refine-resolution-level 2
job3=$(sbatch --parsable "${sbatch_common[@]}" \
  --job-name="pgs-mesh-${job_name}" \
  --output="pgs-recon_mesh_%j_out.txt" \
  --dependency="afterok:${job2}" \
  -p "${part_himem}" --cpus-per-task="${cpus_himem}" --mem="${mem_himem}" \
  --time=1-00:00:00 \
  <<EOF
#!/bin/bash
set -e
module load ccs/singularity

echo "Reconstructing, refining and texturing (reconstruct..texture)"
time singularity run --overlay pgs-recon.overlay "${pgs_recon_container}" \\
  pgs-recon -o "${processed_job_dir}" \\
  --from reconstruct \\
  --threads "\${SLURM_CPUS_PER_TASK}"
EOF
)
printf '  job 3  %-22s %s  %s\n' "reconstruct..texture" "${part_himem}" "${job3}"

# --- Job 4: post-process and deliver ----------------------------------------
# Not part of the staged reconstruction, but kept off the high-memory node so
# the big allocation is not held open for rsync.
job4=$(sbatch --parsable "${sbatch_common[@]}" \
  --job-name="pgs-deliver-${job_name}" \
  --output="pgs-recon_deliver_%j_out.txt" \
  --dependency="afterok:${job3}" \
  --mail-type=END,FAIL \
  -p "${part_cpu}" --cpus-per-task=8 --time=8:00:00 \
  <<EOF
#!/bin/bash
set -e
module load ccs/singularity

echo "Reordering mesh"
time singularity run --overlay registration-toolkit.overlay "${registration_toolkit_container}" \\
  rt_reorder_texture -i "${processed_job_dir}/mvs/${job_name}.obj" \\
  -o "${processed_job_dir}/mvs/${job_name}_reordered.obj"

echo "Centering and scaling mesh"
time singularity run --overlay pgs-recon.overlay "${pgs_recon_container}" \\
  pgs-center \\
  -i "${processed_job_dir}/mvs/${job_name}_reordered.obj" \\
  -o "${processed_job_dir}/${job_name}_final.obj"

echo "Collecting results"
cp "${processed_job_dir}/mvs/${job_name}.obj" \\
  "${processed_job_dir}/mvs/${job_name}.mtl" \\
  "${processed_job_dir}/mvs/${job_name}"_material_*.jpg \\
  "${processed_job_dir}/"

# Package the intermediate files (remove them as we go)
time tar -cvzf "${processed_job_dir}/intermediate.tar.gz" \\
  "${processed_job_dir}/jpg" \\
  "${processed_job_dir}/mvg" \\
  "${processed_job_dir}/mvs" \\
  "${processed_job_dir}/metadata.json" \\
  ${processed_job_dir}/*.txt \\
  --remove-files

echo "Copying results to gemini"
time rsync -a --remove-source-files "${processed_job_dir}" dtn:"${processed_dir}/"

echo "Removing working directory"
rm -rf "${processed_job_dir}" "${root_processing_dir}/${job_name}"

echo "Done."
EOF
)
printf '  job 4  %-22s %s  %s\n' "deliver" "${part_cpu}" "${job4}"

cat <<EOF

Chain submitted for ${job_name}.
  output dir: ${processed_job_dir}

Every dependency is afterok, so a failed job leaves the rest pending and Slurm
eventually cancels them; nothing downstream runs on half-built artifacts.

  squeue -j ${job1},${job2},${job3},${job4}
  scancel ${job1} ${job2} ${job3} ${job4}     # abandon the whole chain

To inspect state without running anything -- loaded arguments, pipeline shape,
which stages are complete, and the prerequisite check:

  singularity run --overlay pgs-recon.overlay ${pgs_recon_container} \\
    pgs-recon -o ${processed_job_dir} --dry-run

Resubmitting any single job unchanged resumes from wherever it died.
EOF
