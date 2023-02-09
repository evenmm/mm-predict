#!/bin/bash
# Open multiple screens, send a command to each screen

screen_names=("test1" "test2" "test3" "test4" "test5" "test6")
job_names=("module purge\n module load GCC/11.2.0\n module load CUDA/11.3.1 \n module load OpenMPI/4.1.1-GCC-11.2.0\n module load Python/3.9.6-GCCcore-11.2.0\n python simdata_BNN.py 1\n" "module purge\n module load GCC/11.2.0\n module load CUDA/11.3.1 \n module load OpenMPI/4.1.1-GCC-11.2.0\n module load Python/3.9.6-GCCcore-11.2.0\n python simdata_BNN.py 2\n" "module purge\n module load GCC/11.2.0\n module load CUDA/11.3.1 \n module load OpenMPI/4.1.1-GCC-11.2.0\n module load Python/3.9.6-GCCcore-11.2.0\n python simdata_BNN.py 3\n" "module purge\n module load GCC/11.2.0\n module load CUDA/11.3.1 \n module load OpenMPI/4.1.1-GCC-11.2.0\n module load Python/3.9.6-GCCcore-11.2.0\n python simdata_BNN.py 4\n" "module purge\n module load GCC/11.2.0\n module load CUDA/11.3.1 \n module load OpenMPI/4.1.1-GCC-11.2.0\n module load Python/3.9.6-GCCcore-11.2.0\n python simdata_BNN.py 5\n" "module purge\n module load GCC/11.2.0\n module load CUDA/11.3.1 \n module load OpenMPI/4.1.1-GCC-11.2.0\n module load Python/3.9.6-GCCcore-11.2.0\n python simdata_BNN.py 6\n")
number_of_jobs=${#screen_names[@]}

for (( job=0; job < $number_of_jobs; job++ ))
do
  screen -dmS "${screen_names[job]}"
  screen -S "${screen_names[job]}" -p 0 -X stuff "${job_names[job]}"
done