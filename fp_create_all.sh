#!/bin/bash

targets=("NR-AR" "NR-AR-LBD" "NR-AhR" "NR-Aromatase" "NR-ER" "NR-ER-LBD" "NR-PPAR-gamma" "SR-ARE" "SR-ATAD5" "SR-HSE" "SR-MMP" "SR-p53")
descriptors=('ecfp0' 'ecfp2' 'ecfp4' 'ecfp6' 'fcfp2' 'fcfp4' 'fcfp6' 'maccs' 'hashap' 'hashtt' 'avalon' 'rdk5' 'rdk6' 'rdk7' "dist_2D" 'dist_3D' 'balaban_2D' 'balaban_3D' 'adjac' 'Laplacian' 'inv_dist_2D' 'inv_dist_3D' 'CMat_full' 'CMat_400' 'CMat_600' 'eigenvals' "rdkit_descr")

cd /auto/brno2/home/nierja/classification/Tox

for target in ${targets[@]}; do
	for fp in ${descriptors[@]}; do
  		qsub -v fp=$fp,target=$target cpu_job.sh
	done
done
