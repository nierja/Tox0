#!/bin/bash
cd /auto/brno2/home/nierja/classification/Tox
targets=("NR-AR" "NR-AR-LBD" "NR-AhR" "NR-Aromatase" "NR-ER" "NR-ER-LBD" "NR-PPAR-gamma" "SR-ARE" "SR-ATAD5" "SR-HSE" "SR-MMP" "SR-p53")
descriptors=('ecfp4_maccs' 'maccs_rdk7' 'ecfp4_rdk7' 'ecfp0' 'ecfp2' 'ecfp4' 'ecfp6' 'fcfp2' 'fcfp4' 'fcfp6' 'maccs' 'hashap' 'hashtt' 'avalon' 'rdk5' 'rdk6' 'rdk7' 'CMat_400' 'CMat_600' 'eigenvals' "rdkit_descr")
pcas=('0' '20' '50')

for target in ${targets[@]}; do
	for fp in ${descriptors[@]}; do
        	for pca in ${pcas[@]}; do
  		    qsub -v fp=$fp,target=$target,pca=$pca hp_job.sh
        	done
	done
done
