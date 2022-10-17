#!/bin/bash
#PBS -N fp_job
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_local=8gb
#PBS -l walltime=0:30:00 
#PBS -q cucam

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/auto/brno2/home/nierja/classification/Tox # substitute username and path to to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

#loads the Gaussian's application modules, version 03
cd $DATADIR
export TMPDIR=$SCRATCHDIR
module add python/3.8.0-gcc-rab6t cuda/cuda-11.2.0-intel-19.0.4-tn4edsz cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t
python3 -m venv GPU_TF_RDKIT
GPU_TF_RDKIT/bin/pip install --no-cache-dir --upgrade pip setuptools
GPU_TF_RDKIT/bin/pip install --no-cache-dir tensorflow==2.8.0 tensorflow-addons==0.16.1 tensorflow-probability==0.16.0 tensorflow-hub==0.12.0 gym==0.20.0 scipy numpy pandas tabulate matplotlib rdkit
source ./GPU_TF_RDKIT/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

python classification.py --target=$target --model=$model

# clean the SCRATCH directory
clean_scratch