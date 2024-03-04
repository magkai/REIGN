#!/bin/bash
#SBATCH -p cpu20
#SBATCH -c 100
#SBATCH --mem=250GB
#SBATCH -t 25:00:00
#SBATCH -o reign_conquer_convquestions_seed.log


source REIGN_ENV/bin/activate



RUN_NAME="qa/CONQUER/results/convquestions/seed/reign_conquer/"
CHECKPOINTQA=$RUN_NAME"checkpoints-QA"
TRAINDATA=$RUN_NAME"augmented_data/"
EVALDATA="data/convquestions/"
GPTEVALDATA="gpt_eval/data/out/conquer/convquestions/"

start_time="$(date -u +%s)"
python qa/CONQUER/QAModel.py --data $TRAINDATA --refcategories "all"  --storepath $CHECKPOINTQA  --epochs 20 > $RUN_NAME"train_qa.log" || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for QA training"

start_time="$(date -u +%s)"
python qa/CONQUER/eval_QA.py --data $EVALDATA --checkpointpath $CHECKPOINTQA --epochs 20 --resultpath $RUN_NAME >> $RUN_NAME"eval_qa.log" || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for QA inference"


start_time="$(date -u +%s)"
python qa/CONQUER/eval_QA.py --data $GPTEVALDATA --checkpointpath $CHECKPOINTQA --test --flag "gpt" --topepoch 20 --resultpath $RUN_NAME >> $RUN_NAME"eval_qa_gpt.log" || exit 1
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo $elapsed "seconds needed for QA inference"




echo "Finished"


