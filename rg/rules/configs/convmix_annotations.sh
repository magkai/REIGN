#!/bin/bash

source REIGN_ENV/bin/activate

RUN_NAME="rg/rules/"
DATATYPE=$1
DATA="data/convmix/"$DATATYPE"_set_ALL.json"
ANNOTATEDDATA=$RUN_NAME"out/annotated_"$DATATYPE"set.json"
INITIALREFS=$RUN_NAME"out/initialRefs_"$DATATYPE"set.json"
REFDATA="rg/BART/in/rg_input_"$DATATYPE".pickle"

python rg/rules/DatasetAnnotator.py --inpath $DATA --outpath $ANNOTATEDDATA --annotatedPreds "frequent_preds.json" #$RUN_NAME"/out/frequent_preds.json"

python rg/rules/createInitialReformulations.py --inpath $ANNOTATEDDATA --outpath $INITIALREFS 

python rg/rules/prepareData.py --inpath $INITIALREFS --outpath $REFDATA

python rg/rules/createMoreRuleRefs.py --inpath $REFDATA --annotateddataset $ANNOTATEDDATA  --outpath $REFDATA 

echo "Finished"


