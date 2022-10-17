#!/bin/bash

# targets=("NR-AR" "NR-AR-LBD" "NR-AhR" "NR-Aromatase" "NR-ER" "NR-ER-LBD" "NR-PPAR-gamma" "SR-ARE" "SR-ATAD5" "SR-HSE" "SR-MMP" "SR-p53")
# models=("most_frequent" "gbt" "lr" "svm" "adalr" "baglr" "badlr" "mlp", "tf_dnn")
targets=("NR-AR") # "NR-AR-LBD" "NR-AhR" "NR-Aromatase" "NR-ER" "NR-ER-LBD" "NR-PPAR-gamma" "SR-ARE" "SR-ATAD5" "SR-HSE" "SR-MMP" "SR-p53")
models=("most_frequent") # "gbt" "lr" "svm" "adalr" "baglr" "badlr" "mlp", "tf_dnn")


cd /auto/brno2/home/nierja/classification/Tox

for target in ${targets[@]}; do
	for model in ${models[@]}; do
  		qsub -v target=$target,model=$model gpu_job.sh
	done
done
