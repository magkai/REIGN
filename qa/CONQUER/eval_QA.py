
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("utils")
import utils as ut
import json


# Set a seed value
seed_value= 12345 # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.agents import ReinforceAgent

from tf_agents.trajectories import time_step as ts


import pickle
import logging
import pdb
import sys

logging.disable(logging.WARNING)

from QAPolicyNetwork import QAPolicyNetwork
from argparse import ArgumentParser


tf.compat.v1.enable_v2_behavior()


train_step_counter = tf.compat.v2.Variable(0)
#use CONQUER default values here
learning_rate = 1e-3
alpha = 0.0
nbr_sample_actions = 5

action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=999, name='action')
observation_spec = array_spec.ArraySpec(shape=(1001,769), dtype=np.float32, name='observation')

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)


actor_network = QAPolicyNetwork( 
    seed_value,
  tensor_spec.from_spec(observation_spec), 
  tensor_spec.from_spec(action_spec))


parser = ArgumentParser()
parser.add_argument("--checkpointpath", dest="checkpoint_path")
parser.add_argument("--checknum", dest="checknum", type=int, default=10)
parser.add_argument("--data", dest="data", default="")
parser.add_argument("--resultpath", dest="resultpath", default="./")
parser.add_argument("--epochs", dest="epochs", default="20")
parser.add_argument("--topepoch", dest="topepoch", default="0")
parser.add_argument("--flag", dest="flag", default="")
parser.add_argument("--ignoreoriginal", action='store_true')
parser.add_argument("--test", action='store_true')
args = parser.parse_args()

epochs = args.epochs
topepoch = args.topepoch
checkpoint_path = args.checkpoint_path
ignoreoriginal = args.ignoreoriginal
test = args.test
resultpath = args.resultpath
flag = args.flag


eval_policy = None


def initialize(checknum):
    checkpoint = tf.train.Checkpoint(actor_net=actor_network)
    checkpoint.restore(checkpoint_path  + "/ckpt-" + str(int(checknum))) #+ str(check_num)) 


    rfAgent = ReinforceAgent(
        tensor_spec.from_spec(ts.time_step_spec(observation_spec)), tensor_spec.from_spec(action_spec), actor_network, optimizer, train_step_counter=train_step_counter
    )
    rfAgent.initialize()

    global eval_policy 
    eval_policy= rfAgent.policy

def initialize_dev(checknum):
    initialize(checknum)
    
    global question_info
    global entity_info
    
    with open(args.data + "question_info_devset.pickle", "rb") as quest_file:
        question_info = pickle.load(quest_file)

    with open(args.data + "entity_info_devset.pickle", "rb") as quest_file:
        entity_info = pickle.load(quest_file)



def initialize_test(checknum):
    initialize(checknum)
    
    global question_info
    global entity_info
    
    with open(args.data + "question_info_testset.pickle", "rb") as quest_file:
        question_info = pickle.load(quest_file)

    with open(args.data + "entity_info_testset.pickle", "rb") as quest_file:
        entity_info = pickle.load(quest_file)


def isExistential(question_start):
    existential_keywords	= ['is', 'are', 'was', 'were', 'am', 'be', 'being', 'been', 'did', 'do', 'does', 'done', 'doing', 'has', 'have', 'had', 'having']
    if question_start in existential_keywords:
        return True
    return False
       
def getPrecisionAt1(answers, goldanswers):
    goldanswers_lower = [ga["id"].lower() for ga in goldanswers]
    for answer in answers:
        if answer[2] > 1:
            return 0.0
        if answer[0].lower() in goldanswers_lower:
            return 1.0
   
    return 0.0

def getHitsAt5(answers, goldanswers):
    goldanswers_lower = [ga["id"].lower() for ga in goldanswers]
    for answer in answers: 
        if answer[2] > 5:
            return 0.0
        if answer[0].lower() in goldanswers_lower:
            return 1.0
    return 0.0

# def getHitsAt50(answers, goldanswers):
#     goldanswers_lower = [ga.lower() for ga in goldanswers]
#     for answer in answers: 
#         if answer[2] > 50:
#             return 0.0
#         if answer[0].lower() in goldanswers_lower:
#             return 1.0
#     return 0.0

def getMRR(answers, goldanswers):
    goldanswers_lower = [ga["id"].lower() for ga in goldanswers]
    i = 0
    for answer in answers:
        if answer[0].lower() in goldanswers_lower:
            return 1.0/answer[2]
        i+=1
    return 0.0


def call_rl(timesteps, start_ids):
    answers = dict()
    
    action_step = eval_policy.action(timesteps) #, policy_state)
    all_actions = np.arange(1000)
    all_actions = tf.expand_dims(all_actions, axis=1)
    distribution = actor_network.get_distribution()

    log_probability_scores = distribution.log_prob(all_actions)
    log_probability_scores = tf.transpose(log_probability_scores)
    top_log_scores, topActions = tf.math.top_k(log_probability_scores,nbr_sample_actions)
 
    for i in range(len(start_ids)):
        for j in range(len(topActions[i])):
            if j == 0:
                answers[start_ids[i]] = []
    
            if not start_ids[i] in entity_info.keys():
                answers[start_ids[i]].append("")
                continue
            paths = entity_info[start_ids[i]]["actions"]
            if topActions[i][j].numpy() >= len(paths):
                answers[start_ids[i]].append("")
                continue
            answerpath = paths[topActions[i][j]]
            if len(answerpath) < 3:
                answers[start_ids[i]].append("")
                continue
            answer = answerpath[2]
            answers[start_ids[i]].append(answer)
          
    return (answers, top_log_scores.numpy())
    
def create_observation(startId, encoded_history, encoded_question):
    if not startId in entity_info.keys() or len(entity_info[startId]["actions"]) == 0 or entity_info[startId]["action_num"] == 0:
        return None
   
    encoded_actions = entity_info[startId]["encoded_actions"] #[:1000]
    action_nbr = entity_info[startId]["action_num"]
  
    mask = tf.ones(action_nbr)
    if encoded_history is None:
        zeros = tf.zeros((1001-action_nbr))
    else:
        zeros = tf.zeros((1002-action_nbr))
    mask = tf.keras.layers.concatenate([mask, zeros], axis=0)
    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1)#[1,2003,1]
 
    if encoded_history is None:
        observation = tf.keras.layers.concatenate([encoded_question, encoded_actions],axis=0) #[2003, 768]
    else:
        observation = tf.keras.layers.concatenate([encoded_history, encoded_question, encoded_actions],axis=0) #[2003, 768]
  
    observation = tf.expand_dims(observation, 0) #[1, 2003, 768]
    observation =  tf.keras.layers.concatenate([observation, mask], axis=2) #[1,2003,769]
    tf.dtypes.cast(observation, tf.float32)
    

    return observation


def findAnswer(currentid):
    
    
    currentStarts = question_info[currentid]["entities"]
    if len(currentStarts) == 0:
        return []
  
    timeSteps = None
    observations = None
    startIds = []
  
    for startNode in currentStarts:
    
        startid = startNode
        observation = create_observation(startid,None,question_info[currentid]["encoded_question"])
        if observation is None:
            continue
        if observations is None:
            observations = observation
        else:
            observations = np.concatenate((observations, observation))
        startIds.append(startNode)
   
    if not observations is None:
     
        timeSteps = ts.restart(observations, batch_size=observations.shape[0])
  
    if not timeSteps is None:
        #print("timeSteps: ", timeSteps)
        answers, log_probs = call_rl(timeSteps, startIds)
    else:
        return []
    
    i = 0
    additive_answer_scores = dict()
  
    for sId in startIds:
        for j in range(len(log_probs[i])):
            score = np.exp(log_probs[i][j])
            curr_answer = answers[sId][j]
            if curr_answer == "":
              #  print("empty answer for startid: ", sId[0], "with log prob: ", log_probs[i][j])
                continue
         
            if curr_answer in additive_answer_scores.keys():
                additive_answer_scores[curr_answer] += score
            else: 
                additive_answer_scores[curr_answer] = score
               
        i +=1

   
    additive_answer_scores = sorted(additive_answer_scores.items(), key=lambda item: item[1], reverse=True)
    return additive_answer_scores


def calculateAnswerRanks(answer_scores):
    if len(answer_scores) == 0:
        return []
    rank = 0
    same_ranked = 0
    prev_score = ["", ""]
   
    for score in answer_scores:
        if score[1] == prev_score[1]:
            same_ranked +=1 
        else:
            rank += (1 + same_ranked)
            same_ranked = 0
        score.append(rank)
        prev_score = score
    return answer_scores


def formatAnswer(answer):
    if len(answer) == 0:
        return answer
    best_answer = answer
    if answer[0] == "Q" and "-" in answer:
        best_answer = answer.split("-") [0]
    #elif ut.is_timestamp(answer):
     #   best_answer = ut.convertTimestamp(answer)
 
    return best_answer



def doEval(epoch):

    avg_prec = 0.0
    avg_hits_5 = 0.0
    avg_mrr = 0.0
     
    question_count = 0
    existential_count = 0
    total_answered = 0
    total_exits_answered = 0
    answer_dict = dict()
    

    for qid in question_info.keys():
        if ignoreoriginal and qid.endswith("-0-0"):
            continue
        convId = qid.split("-")[0] + "-" + qid.split("-")[1]
        question_count += 1
        question = question_info[qid]["question"]
        question_start = question.split(" ")[0].lower()
        origId = qid.split("-")[0] + "-" + qid.split("-")[1] + "-0-0"
        gold_answers = question_info[origId]["answers"]
    

        if isExistential(question_start):
            existential_count += 1
            # always Yes:
            answer_scores = [["Yes", 1.0 , 1], ["No", 0.5 ,2]]
        
        else:
            answer_scores = findAnswer(qid)
            answer_scores = [list(a) for a in answer_scores]             
            
            for answer in answer_scores:
                answer[1] = format(answer[1],  '.3f')
                answer[0] = formatAnswer(answer[0]) 
            calculateAnswerRanks(answer_scores)

    
        if not convId in answer_dict.keys():
            answer_dict[convId] = dict()
            answer_dict[convId]['conv_id'] = convId
            answer_dict[convId]["questions"] = []

        newEntry = dict()
        newEntry["question"] = question
        newEntry["question_id"] = qid
        newEntry["turn"] = int(qid.split("-")[1])
        newEntry["answers"] = gold_answers

        newEntry["ranked_answers"] = []
        for ans in answer_scores:
            newEntry["ranked_answers"].append({"answer": {"id": ans[0], "label": ut.getLabel(ans[0])} , "score": ans[1], "rank": ans[2]})
        
    
        prec = getPrecisionAt1(answer_scores, gold_answers)
        avg_prec += prec
        hits_5 = getHitsAt5(answer_scores, gold_answers)
        #hits_50 = getHitsAt50(answer_scores, gold_answers)
        avg_hits_5 += hits_5
        #avg_hits_50 += hits_50
        mrr = getMRR(answer_scores,gold_answers)
        avg_mrr += mrr
        newEntry["p_at_1"] = prec
        newEntry["h_at_5"] = hits_5
        newEntry["mrr"] = mrr
      
        if prec == 1.0:
            total_answered += 1
            if isExistential(question_start):
                total_exits_answered += 1

        answer_dict[convId]["questions"].append(newEntry)

    answerList = list(answer_dict.values())
   # with open(resultfile_path + "gold_answers.json", "w") as train_out:
    #    json.dump(answerList, train_out)
    print("EPOCH :" + str(epoch)+ "\n")
    print("AVG results: " + str(format(avg_prec/question_count, '.3f')) + "," +  str(format(avg_mrr/question_count, '.3f')) +   "," + str(format(avg_hits_5/question_count, '.3f')) +  "(P@1, MRR, HIT@5)\n" )
    print("total number of information needs: "+ str(question_count)+ "\n" )
    print("total number of correctly answered questions: "+ str(total_answered)+ "\n" )
    print("----------------------------------------------------------------------------------------------\n")       
    return answerList, avg_prec    




if __name__ == '__main__':
       
  
    global_best_mrr = 0.0
    if int(topepoch) > 0:
        if test:
            print("in test")
            initialize_test(int(topepoch))
        else:
            initialize_dev(int(topepoch))
        answerList, avg_prec = doEval(int(topepoch))
        if flag == "":
           name = "gold_answers.json" 
        else:
            name = "gold_answers_" + flag + ".json"
        with open(resultpath + name, "w") as train_out:
            json.dump(answerList, train_out)
    else:
        global_best = 0.0
        for epoch in range(5,int(epochs)+1):
            initialize_dev(int(epoch))
            answerList, avg_prec = doEval(int(epoch))
            if avg_prec > global_best:
                global_best = avg_prec
                best_epoch = epoch
                print("new global best: ", avg_prec, "for epoch: ", epoch)
                with open(resultpath + "gold_answers_dev.json", "w") as train_out:
                    json.dump(answerList, train_out)
        
        initialize_test(int(best_epoch))
        print("best epoch: ", best_epoch)
        answerList, avg_prec = doEval(int(topepoch))
        with open(resultpath + "gold_answers.json", "w") as train_out:
            json.dump(answerList, train_out)
