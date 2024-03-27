#!/usr/bin/bash 

mkdir -p models
mkdir -p data

# download reign convmix data + convquestions seed data
wget http://qa.mpi-inf.mpg.de/reign/data/data.zip
unzip data.zip 
rm data.zip

#get kg labels
cd data
mkdir -p kg
cd kg
wget http://qa.mpi-inf.mpg.de/reign/data/labels.json
cd ../../

#download gpt test data processed
wget http://qa.mpi-inf.mpg.de/reign/data/gpt_data.zip
unzip gpt_data.zip 
rm gpt_data.zip

#download RG distant supervision data + results:
cd rg/BART/
wget http://qa.mpi-inf.mpg.de/reign/data/rg/rg_data.zip
unzip rg_data.zip 
rm rg_data.zip
cd ../../
#TODO: add rule-based data

#download RCS data + results:
cd rcs
wget http://qa.mpi-inf.mpg.de/reign/data/rcs/results.zip
unzip results.zip 
rm results.zip
cd ../


#download CONQUER training results:
cd qa/CONQUER/
wget http://qa.mpi-inf.mpg.de/reign/data/qa/conquer/results.zip
unzip results.zip 
rm results.zip
cd ../../

#download EXPLAIGNN training results:
cd qa/EXPLAIGNN/
mkdir -p _results
wget http://qa.mpi-inf.mpg.de/reign/data/qa/explaignn/_results.zip
unzip _results.zip 
rm _results.zip

mkdir -p _data
wget http://qa.mpi-inf.mpg.de/reign/data/qa/explaignn/_data.zip
unzip _data.zip 
rm _data.zip

mkdir -p _intermediate_representations
wget http://qa.mpi-inf.mpg.de/reign/data/qa/explaignn/_intermediate_representations.zip
unzip _intermediate_representations.zip 
rm _intermediate_representations.zip


echo "Successfully downloaded data!"
