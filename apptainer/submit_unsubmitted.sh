#!/bin/bash

cnt=0
for i in $(cat "$1"); do 
  if grep -Fxq "$i" "submitted.txt"; then
    continue
  fi

  # Check if we've already submitted 10 jobs
  if (($cnt == 10)); then
     echo "Submitted ${cnt} jobs. Delaying 2 mins to avoid overloading ssh..."
     sleep 120
     echo "Resuming..."
     echo
     cnt=0
  fi

  # Submit a new chain. submit_recon_pipeline.sh runs on the login node and
  # sbatches the jobs itself, so it is invoked directly; per-job partitions and
  # resources are set inside it (PGS_PART_* to override).
  echo "Submitting ${i}..."
  PGS_EXPOSURE=2 PGS_SHADOWS=40 ./submit_recon_pipeline.sh "/mnt/gemini1-4/seales_uksr/herculaneum/Dailies/${i}/"
  echo ${i} >> submitted.txt
  echo

  cnt=$(($cnt+1))
done

