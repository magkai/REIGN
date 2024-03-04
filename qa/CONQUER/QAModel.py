import json
import pickle
import tensorflow as tf
import logging
import numpy as np
import sys
import pdb
import requests
import threading
import wandb
import copy
from argparse import ArgumentParser

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#tf.debugging.set_log_device_placement(True)
logging.disable(logging.WARNING)


sys.path.append("utils")

from QAPolicyNetwork import QAPolicyNetwork
from tf_agents.trajectories import Trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec

from tf_agents.trajectories import time_step_spec
from tf_agents.agents import ReinforceAgent


import utils as ut

import random

random.seed(12345)
avg_rewardList = []

class QAModel():

    def __init__(self, ref_file,data, number_of_rollouts, batch_size, learning_rate, store_path,load_path, check_num, refCategories, dataflag, only_refs=False, refPerEpoch=False):
        action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=999, name='action')
        observation_spec = array_spec.ArraySpec(shape=(1001,769), dtype=np.float32, name='observation')

        self.refPerEpoch = refPerEpoch
        self.seed_value = 12345
        entropy_const = 0.1
        learning_rate = learning_rate
        self.batch_size = batch_size
        self.rollouts = number_of_rollouts
        print("learning rate: ", learning_rate)
        print("batch size: ", self.batch_size)
        print("rollouts: ", self.rollouts)
        self.top_k_num = 5
        #self.epoch_nbr = epoch_nbr
        self.check_num = check_num
        self.ref_file = ref_file        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        tf.compat.v1.enable_v2_behavior()
        train_step_counter = tf.compat.v2.Variable(0)
        
      
        self.QANet = QAPolicyNetwork(self.seed_value,tensor_spec.from_spec(observation_spec), tensor_spec.from_spec(action_spec))
        
        self.QAReinforceAgent = ReinforceAgent(
            time_step_spec(tensor_spec.from_spec(observation_spec)), 
            tensor_spec.from_spec(action_spec), 
            self.QANet, 
            optimizer, 
            entropy_regularization=entropy_const, 
            #normalize_returns=False,
            train_step_counter=train_step_counter)
        self.QAReinforceAgent.initialize()
        self.collect_policy = self.QAReinforceAgent.collect_policy
       
         #create checkpoint for weights of the policy network
        self.checkpoint = tf.train.Checkpoint(actor_net=self.QANet)
        self.storePath = store_path
        self.loadPath = load_path
        
        if dataflag != "":
            
            with open(data + "question_info_trainset_" + dataflag + ".pickle" , "rb") as infile:
                self.question_info = pickle.load(infile)

            with open(data + "entity_info_trainset_" + dataflag + ".pickle" , "rb") as infile:
                self.ent_info = pickle.load(infile)
        else:
            with open(data + "question_info_trainset.pickle" , "rb") as infile:
                self.question_info = pickle.load(infile)

            with open(data + "entity_info_trainset.pickle" , "rb") as infile:
                self.ent_info = pickle.load(infile)

        if refCategories == "all":
            self.refCategories = "all"
        elif refCategories == "orig_augment":
            self.refCategories = "orig_augment"
        else:
            self.refCategories = refCategories.split(",")
       
        print("refcats; ", self.refCategories)
        self.refExperience = dict()
        self.final_obs = False
        self.question_counter = 0
        self.LAST = False
        self.onlyRefs = only_refs

   
    
    def filterRefCategory(self):
        new_data = dict()
        for qId in self.question_info.keys():
            if "ref_category" in self.question_info.keys():
                ref_cat = self.question_info[qId]["ref_category"]
            else:
                ref_cat = qId.split("-")[2]
            
            if str(ref_cat) in self.refCategories:
                new_data[qId] = self.question_info[qId]
        
        self.question_info = new_data
        print("refs filtered, new length: ", len(self.question_info.keys()))
        return

      
   
    def augmentOriginalQuestion(self):
        new_data = dict()
        for qId in self.question_info.keys():
            if qId.endswith("0-0"):
                new_data[qId] = self.question_info[qId]
                for i in range(5):
                    new_data[qId[:-2] + "-" + str(i+1)] = self.question_info[qId]
        self.question_info = new_data
        return


    def reset(self,i):
        self.final_obs = False
        self.question_counter = 0
        self.refExperience = dict()
        if self.refCategories == "orig_augment":
            self.augmentOriginalQuestion()
        elif self.refCategories != "all":
            self.filterRefCategory()
       
        self.createBatchedData()
      
        return
        
    
    def createBatchedData(self):
        self.batchedData = []
        qCount = 0
        entCount = 0
        for qId in self.question_info.keys():
            if self.onlyRefs:
                if not qId.endswith("0-0"):
                    continue 
            qCount += 1
            for ent in self.question_info[qId]["entities"]:
                self.batchedData.append([qId, ent])
                entCount += 1
        random.shuffle(self.batchedData)
        print("number of train questions: ", qCount, " ; number of entities: ", entCount)
        return 

   

    def cycle(self, last=False):
        if last:
            self.LAST = True
        iter = 0
        while not self.final_obs:
           
            observation, obsTuple, refList = self.createBatchedObservation()
            
            if not observation is None:
                traj = self.createBatchedExperienceWithRollouts(observation, obsTuple, refList)
                self.updateModel(traj, iter)
            iter += 1

        return

  
    def _getActionMask(self, action_nbr):
        mask = tf.ones(action_nbr)
        zeros = tf.zeros((1001-action_nbr))
        mask = tf.keras.layers.concatenate([mask, zeros], axis=0)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1)#[1,1001,1] or [1,1002,1]
        return mask

    def createBatchedObservation(self):
  
        if self.question_counter + self.batch_size > len(self.batchedData)-1:
            print("end of training samples: empty observation returned")
            self.final_obs = True
         #   self.idLock.release()
            return None, None, None
    
    
        batch = self.batchedData[self.question_counter:self.question_counter+self.batch_size]
        self.question_counter+= self.batch_size
     
        refList = [] 
        obsTuple = []
        encoded_questions = None
        encoded_actions = None
        action_masks = None
        
        for qId, ent in batch:
            if not ent in self.ent_info.keys():
                print("WARNING: entity not in dict, potential error for ent: ", ent, ", qid: ", qId)
                continue
        
            refList.append(qId) 
            enc_quest = self.question_info[qId]["encoded_question"]
            action_num = self.ent_info[ent]["action_num"]
            if action_num == 0:
                print("no action for entity: ", ent, " and qid: ", qId)
                continue

            if encoded_questions is None:
                encoded_questions = enc_quest
            else:
                encoded_questions = tf.keras.layers.concatenate([encoded_questions,  enc_quest], axis=0)
            
            enc_acts = self.ent_info[ent]["encoded_actions"]
           
            action_mask = self._getActionMask(action_num)
            if action_masks is None:
                action_masks = action_mask
            else:
                action_masks = tf.keras.layers.concatenate([action_masks, action_mask], axis=0)
            
            enc_acts = tf.expand_dims(enc_acts, 0)
            if encoded_actions is None:
                encoded_actions = enc_acts
            else:
                encoded_actions = tf.keras.layers.concatenate([encoded_actions, enc_acts], axis=0)
    
            obsTuple.append((qId, ent))
            
        if encoded_questions is None:
            return None, None, None
        encoded_questions = tf.expand_dims(encoded_questions, axis=1)
     
        observations = None
        observations = tf.keras.layers.concatenate([encoded_questions, encoded_actions],axis=1) #[1001, 768]
       
        observations =  tf.keras.layers.concatenate([observations, action_masks], axis=2) #[1,1001,769] (or [1, 1002, 769])
        observations = tf.dtypes.cast(observations, tf.float32)
        
        return observations, obsTuple, refList



    def getQARewardsWithRollouts(self, obsTuple, refList, actions, sampledActions, topActions):
        
        rewards = []
        idList = []
        allActions = []
        
        for i in range(len(actions)):
            qId, ent = obsTuple[i] 
            #check selected action 
            try:
                
                answer = self.ent_info[ent]["actions"][actions[i].numpy()][2]
                #if ut.is_timestamp(answer):
                 #   answer = ut.convertTimestamp(answer)
            except Exception:
                answer = None
                #print("Error, cannot find path for entity ", ent, " and action ", actions[i])
            
            allActions.append(actions[i])
            if qId.endswith("-0-0"):
                orig_qId = qId
            else:
                splitted = qId.split("-")
                orig_qId = splitted[0] + "-" + splitted[1] + "-0-0"
            if orig_qId in self.question_info.keys():
                gas = [ans["id"] for ans in self.question_info[orig_qId]["answers"] ]   
            else:
                gas = [ans["id"] for ans in self.question_info[qId]["answers"] ]   
            if answer in gas:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
 
            #check sampled actions
            for j in range(len(sampledActions[i])):
                allActions.append(sampledActions[i][j])
                try:
                    answer =self.ent_info[ent]["actions"][sampledActions[i][j].numpy()][2]
                    #if ut.is_timestamp(answer):
                     #   answer = ut.convertTimestamp(answer)
                except Exception:
                    answer = None
                   
                if answer in gas:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)

            #check top-k greedy actions
            if self.LAST:
                for k in range(self.top_k_num):
                    try:
                        answer =self.ent_info[ent]["actions"][topActions[i][k].numpy()][2]
                        #if ut.is_timestamp(answer):
                         #   answer = ut.convertTimestamp(answer)
                    except Exception:
                        answer = None
                        print("Error, cannot find path for entity ", ent, " and action ", topActions[i][k],  ", number of available actions was: ", self.number_of_actions[ent])
                    
                    if answer in gas:
                        if qId in refList:   
                            self.refExperience[qId] = 1.0
                            idList.append(qId)
                            break
                                 

                for qId in refList:
                    if not qId in idList:
                        self.refExperience[qId] = 0.0
                       

        avg_rewardList.extend(rewards)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        allActions = tf.convert_to_tensor(allActions, dtype=tf.int32)
        return rewards, allActions
    
   

    def createBatchedExperienceWithRollouts(self, observations, obsTuple, refList):
      
        observations = tf.expand_dims(observations, axis=0)
        
        time_step = ts.restart(observations)
        action_step = self.collect_policy.action(time_step, seed=self.seed_value)
        actions = action_step.action
        distribution = self.QANet.get_distribution()

        all_actions = np.arange(1000)
        all_actions = tf.expand_dims(all_actions, axis=1)
        log_probability_scores = distribution.log_prob(all_actions)
       
        log_probability_scores = tf.transpose(log_probability_scores)
        _, topActions = tf.math.top_k(log_probability_scores,self.top_k_num)
    
        sampledActions =  tf.nest.map_structure(
            lambda d: d.sample((self.rollouts-1), seed=self.seed_value),
            distribution)
      
        sampledActions = tf.transpose(sampledActions)
        rewards, allActions = self.getQARewardsWithRollouts(obsTuple, refList, actions, sampledActions, topActions)
        rewards = tf.expand_dims(rewards, axis=0)
        batch_size = len(actions)*self.rollouts
        allActions = tf.expand_dims(allActions, axis=0)
        observations = tf.repeat(observations,[self.rollouts], axis=1)
        discounts = tf.zeros([1,batch_size], dtype=tf.float32)
        step_type = tf.zeros([1,batch_size], dtype=tf.int32)
        next_step_type =  tf.repeat(2, [batch_size])
        next_step_type = tf.expand_dims(next_step_type, axis=0)
        traj = Trajectory(step_type, observations, allActions, (), next_step_type, rewards, discounts)
      
        return traj

   

    def updateModel(self, traj, iter):
        avg_rewards = tf.reduce_mean(traj.reward).numpy()
        train_loss = self.QAReinforceAgent.train(traj)
        qa_size = traj.reward.shape[1]
        print("QA iteration ", iter, ": avg_rewards = ", avg_rewards, ", loss = ", train_loss.loss.numpy(), ", batchsize of " , qa_size, flush=True)
       
       

    def saveModel(self):
        #save checkpoints for each epoch
        self.checkpoint.save(self.storePath + "/ckpt")
        if not self.ref_file == "":
            with open(self.ref_file, "w") as qfile:
                json.dump(self.refExperience, qfile)


        
    def loadModel(self):
        self.checkpoint.restore(self.loadPath + "/ckpt-" + str(self.check_num)).expect_partial()
       

if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument("config")
    parser.add_argument("--data", dest="data")
    parser.add_argument("--dataflag", dest="dataflag", default="")
    parser.add_argument("--refcategories", dest="ref_categories", default="0")
    parser.add_argument("--epochs", dest="epoch_nbr", type=int, default=10)
    parser.add_argument("--refPerEpoch", action='store_true')
    parser.add_argument("--rollouts", dest="number_of_rollouts", type=int, default=20)
    parser.add_argument("--batchsize", dest="batch_size", type=int, default=50)
    parser.add_argument("--learningrate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--storepath", dest="store_path")
    parser.add_argument("--loadpath", dest="load_path", default="")
    parser.add_argument("--reffile", dest="ref_file", default="")
    parser.add_argument("--onlyrefs", dest="only_refs", default=False)
    #parser.add_argument("--questioninfo", dest="question_info")
    parser.add_argument("--checknum", dest="check_num")
    #parser.add_argument("--entityinfo", dest="entity_info")
    args = parser.parse_args()

    QA = QAModel(args.ref_file, args.data, args.number_of_rollouts, args.batch_size, args.learning_rate, args.store_path, args.load_path,  args.check_num, args.ref_categories, args.dataflag, args.only_refs, args.refPerEpoch)
   
 
    print("number of epochs: ", args.epoch_nbr)
    for i in range(int(args.epoch_nbr)):
     
        if i == 0 and args.load_path != "":
            QA.loadModel()
        QA.reset(i)
        if i == int(args.epoch_nbr)-1 and args.ref_file != "":
            QA.cycle(True)
        else:
            QA.cycle()
        finalAvgReward = np.mean(avg_rewardList)
        print("avg QA rewards in epoch", i+1, ": ", finalAvgReward)
        avg_rewardList = []
        #if i==0 or i%5==0:
        QA.saveModel()
        i+=1