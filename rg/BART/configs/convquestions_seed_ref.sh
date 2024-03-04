#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 10
#SBATCH --mem=250GB
#SBATCH -t 45:00:00
#SBATCH --gres gpu:1
#SBATCH -o logs/convquestions_seed_ref.log


cuda_version=11.3
export PATH=/usr/lib/cuda-${cuda_version}/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/lib/cuda-${cuda_version}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_PATH=/usr/lib/cuda-${cuda_version}/

source REIGN_ENV/bin/activate


RUN_NAME="rg/BART/"
DATATYPE=$1
FILENAME=$RUN_NAME"out/convquestions/seed/question_info_"$DATATYPE"set.pickle"


start_time="$(date -u +%s)"
python rg/BART/generateReformulations.py --refoption "onepercategory"  --checkpointpath $RUN_NAME"checkpoint/" --checknum 3 --filename  "data/convquestions/seed/question_info_"$DATATYPE"set.pickle" --storepath $FILENAME  > "genRefs_convquestions_seed.log"
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for ref generation"

start_time="$(date -u +%s)"
python preprocessing/entitiesRetrieval.py --filename  $FILENAME 
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for entities retrieval"

start_time="$(date -u +%s)"
python preprocessing/encodeQuestions.py --filename  $FILENAME
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for question encoding"

start_time="$(date -u +%s)"
python preprocessing/pathRetrievalEncoding.py --filename  $FILENAME --entityfilename $RUN_NAME"out/seed/convquestions/entity_info_"$DATATYPE"set.pickle" --datatype $DATATYPE
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for path retrieval"



echo "Finished"

