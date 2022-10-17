#!/bin/bash
#PBS -N fp_job
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=8gb
#PBS -l walltime=1:00:00 

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/auto/brno2/home/nierja/classification/Tox # substitute username and path to to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

#loads the Gaussian's application modules, version 03
cd $DATADIR
export TMPDIR=$SCRATCHDIR
module add python/3.8.0-gcc-rab6t
python3 -m venv RDKIT
RDKIT/bin/pip install --no-cache-dir --upgrade pip setuptools
RDKIT/bin/pip install --no-cache-dir scipy numpy pandas tabulate matplotlib rdkit sklearn
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source ./RDKIT/bin/activate

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

python create_training_data.py --fp=$fp --target=$target

# clean the SCRATCH directory
clean_scratch