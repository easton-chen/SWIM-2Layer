#!/bin/bash
cd $1
scaFiles=$(find . -name "*.sca")
scaFileArray=(${scaFiles// / })
for file in ${scaFileArray[@]}
do
	
	echo "scavetool x ${file%.*}.sca -o ${file%.*}.csv"
	echo "scavetool x ${file%.*}.vec -o ${file%.*}.csv"
	scavetool x ${file%.*}.sca -o ${file%.*}.csv
	scavetool x ${file%.*}.vec -o ${file%.*}.csv
done
rm -rf csv
mkdir csv
mv *.csv csv

cd -

#python plotResult.py
