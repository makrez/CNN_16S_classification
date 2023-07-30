#!/bin/bash

for i in $(ls | grep Actinobacteria | grep ne_50); do
		f1=$(cat  ${i}/model_evaluation/f1_classification_report.txt | grep -A 1 'F1 Score' | tail -n +2)
		echo ${i}","${f1}


done
