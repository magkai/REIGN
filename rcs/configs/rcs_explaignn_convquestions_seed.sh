#!/bin/bash
source REIGN_ENV/bin/activate


RUN_NAME="rcs/results/convquestions/seed/reign_explaignn/"

REWARD=$1
EVALFILENAME="rg/BART/out/convquestions/seed/question_info_trainset.pickle"
CHECKPOINTAS=$RUN_NAME"checkpoints-AS/"
CHECKPOINTASLOAD="rcs/results/convmix/reign_explaignn/checkpoints-AS/" #load rcs trained on convmix and evaluate on convquestions only
TMPOUT=$RUN_NAME"augmented_data/"
OUTDATA="qa/EXPLAIGNN/_intermediate_representations/convquestions/reign_explaignn_seed/"


start_time="$(date -u +%s)"
python rcs/RCSModelBase.py --evalonly --addoriginalquestions --samplesize 5 --evalquestioninfo $EVALFILENAME --storequestioninfo $TMPOUT"question_info_trainset.pickle"  --loadpath $CHECKPOINTASLOAD --storepath $CHECKPOINTAS  --epochs 5  > $RUN_NAME"train_as_"$REWARD".log" || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for AS training"

start_time="$(date -u +%s)"
python qa/EXPLAIGNN/reign_data_prep.py --inputfile $TMPOUT"question_info_trainset.pickle" --outputfile $OUTDATA"annotated_train.json"   || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for preparing data for explaignn"

CURRENTLOC=$(pwd)
ln -s $CURRENTLOC"/qa/EXPLAIGNN/_benchmarks/convquestions/dev_set_ALL.json" $CURRENTLOC"/"$OUTDATA"annotated_dev.json"

ln -s $CURRENTLOC"/qa/EXPLAIGNN/_benchmarks/convquestions/test_set_ALL.json" $CURRENTLOC"/"$OUTDATA"annotated_test.json"

echo "Finished"