#!/bin/bash
source REIGN_ENV/bin/activate



RUN_NAME="rcs/results/convmix/reign_conquer/"

QAEVALDATA="qa/CONQUER/results/convmix/conquer/initialQA_gold_answers.json"
REWARD=$1
REFFILE=$RUN_NAME$REWARD"_reward.json"
CHECKPOINTAS=$RUN_NAME"checkpoints-AS/"
FILENAME="rg/BART/out/convmix/question_info_devset.pickle"
EVALFILENAME="rg/BART/out/convmix/question_info_trainset.pickle"
OUTDATA="qa/CONQUER/results/convmix/reign_conquer/augmented_data/"

start_time="$(date -u +%s)"
python rcs/RewardCollector.py --inputfile $QAEVALDATA --outputfile $REFFILE --rewardtype $REWARD  || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for reward collection"


start_time="$(date -u +%s)"
python rcs/RCSModelBase.py --actionmask --addoriginalquestions --samplesize 5 --questioninfo $FILENAME --evalquestioninfo $EVALFILENAME --storequestioninfo $OUTDATA"question_info_trainset.pickle"  --reffile $REFFILE --storepath $CHECKPOINTAS  --epochs 5  > $RUN_NAME"train_as_"$REWARD".log" || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for AS training"

CURRENTLOC=$(pwd)
ln -s $CURRENTLOC"/rg/BART/out/convmix/entity_info_trainset.pickle" $CURRENTLOC"/"$OUTDATA"entity_info_trainset.pickle"

echo "Finished"