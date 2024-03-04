#!/bin/bash

source REIGN_ENV/bin/activate



RUN_NAME="rcs/results/convquestions/seed/reign_conquer/"


REWARD=$1
CHECKPOINTAS=$RUN_NAME"checkpoints-AS/"
CHECKPOINTASLOAD="rcs/results/convmix/reign_conquer/checkpoints-AS/" #load rcs trained on convmix and evaluate on convquestions only
EVALFILENAME="rg/BART/out/convquestions/seed/question_info_trainset.pickle"
OUTDATA="qa/CONQUER/results/convquestions/seed/reign_conquer/augmented_data/"


start_time="$(date -u +%s)"
python rcs/RCSModelBase.py --evalonly --addoriginalquestions --samplesize 5 --evalquestioninfo $EVALFILENAME --storequestioninfo $OUTDATA"question_info_trainset.pickle"  --loadpath $CHECKPOINTASLOAD --storepath $CHECKPOINTAS  --epochs 5  > $RUN_NAME"train_as_"$REWARD".log" || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for AS training"

CURRENTLOC=$(pwd)
ln -s $CURRENTLOC"/rg/BART/out/convquestions/entity_info_trainset.pickle" $CURRENTLOC"/"$OUTDATA"entity_info_trainset.pickle"


echo "Finished"