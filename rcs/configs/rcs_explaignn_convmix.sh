#!/bin/bash
source REIGN_ENV/bin/activate


RUN_NAME="rcs/results/convmix/reign_explaignn/"

QAEVALDATA="qa/EXPLAIGNN/_intermediate_representations/convmix/explaignn_rewarding/sr_reign/rers/kb/explaignn/res_explaignn_rewarding_gold_answers.json"

REWARD="rrd"
REFFILE=$RUN_NAME$REWARD"_reward.json"
CHECKPOINTAS=$RUN_NAME"checkpoints-AS/"
FILENAME="rg/BART/out/convmix/question_info_devset.pickle"
EVALFILENAME="rg/BART/out/convmix/question_info_trainset.pickle"
TMPOUT=$RUN_NAME"augmented_data/question_info_trainset.pickle"
OUTDATA="qa/EXPLAIGNN/_intermediate_representations/convmix/reign_explaignn/"



start_time="$(date -u +%s)"
python rcs/RewardCollector.py --inputfile $QAEVALDATA --outputfile $REFFILE --rewardtype $REWARD  || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for reward collection"


start_time="$(date -u +%s)"
python rcs/RCSModelBase.py --actionmask --addoriginalquestions --samplesize 5 --questioninfo $FILENAME --evalquestioninfo $EVALFILENAME --storequestioninfo $TMPOUT  --reffile $REFFILE --storepath $CHECKPOINTAS  --epochs 5  > $RUN_NAME"train_as_"$REWARD".log" || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for AS training"

start_time="$(date -u +%s)"
python qa/EXPLAIGNN/reign_data_prep.py --inputfile $TMPOUT --outputfile $OUTDATA"annotated_train.json"   || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for preparing data for explaignn"

CURRENTLOC=$(pwd)
ln -s $CURRENTLOC"/qa/EXPLAIGNN/_benchmarks/convmix/dev_set/dev_set_ALL.json" $CURRENTLOC"/"$OUTDATA"annotated_dev.json"

ln -s $CURRENTLOC"/qa/EXPLAIGNN/_benchmarks/convmix/test_set/test_set_ALL.json" $CURRENTLOC"/"$OUTDATA"annotated_test.json"

echo "Finished"