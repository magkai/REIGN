

source /GW/qa3/work/reign/ENV/bin/activate

RUN_NAME="qa/CONQUER/results/convmix/conquer/"
CHECKPOINTQA=$RUN_NAME"checkpoints-QA"
EVALDATA="rg/BART/out/convmix/"


python qa/CONQUER/eval_QA.py --data $EVALDATA --checkpointpath $CHECKPOINTQA --topepoch 15 --resultpath $RUN_NAME"/initialQA_epoch15_" > $RUN_NAME"eval_initial_epoch15.log" 


echo "Finished"


