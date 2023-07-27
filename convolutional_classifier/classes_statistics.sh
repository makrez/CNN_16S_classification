#!/bin/bash

classification_levels=("Kingdom" "Phylum" "Order" "Family" "Genus" "Species")
min_samples=(10 20)

for class_level in "${classification_levels[@]}"; do
    for m in "${min_samples[@]}"; do
        echo "Processing: Class Level = $class_level, Min Samples = $m"
        python data_processing_functions.py /scratch/mk_cas/full_silva_dataset/sequences/full_labels_conform.pkl Domain Bacteria "$class_level" "$m" 1 data_descriptions/.
        echo "Completed: Class Level = $class_level, Min Samples = $m"
    done
done
