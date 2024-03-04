import json
import pickle
import argparse

import numpy as np
import tensorflow as tf
import sys
import logging
from argparse import ArgumentParser
from pathlib import Path

logging.disable(logging.WARNING)


from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step_spec
from tf_agents.trajectories import Trajectory
from tf_agents.utils import common
from tf_agents.networks import sequential
sys.path.append("utils")
import random
random.seed(12345)
from RCSDqnAgent import RCSDqnAgent


avg_rewardList = []


class RCSModelBase():

    def __init__(self, question_info_file, eval_question_info_file, refExpPath,loadPath, storePath, check_num, learn_rate, batch_size, question_info_path, structured, action_mask, action_selection, sample_size, multi_step, add_original_questions):
       
        #for training: file with rewards for reformulation and question data file are required
        if refExpPath != "":
            with open(refExpPath, "r") as qfile:
                self.refExperience = json.load(qfile)
        else:
            self.refExperience = dict()
        
        if question_info_file != "":
            with open(question_info_file , "rb") as infile:
                self.question_info = pickle.load(infile)
        
        if eval_question_info_file != "":
            with open(eval_question_info_file, "rb") as qfile:
                self.eval_question_info = pickle.load(qfile)
        
       
        self.seed_value = 12345   
        learning_rate = learn_rate
        self.structured = structured
        self.multi_step = multi_step
        self.actionSelection = action_selection
        self.sampleSize = sample_size
        self.addOriginalQuestions = add_original_questions
        self.action_mask = action_mask

       
        self.questionIds = []
        #we will only go through questions for which we successfully collected rewards
        if refExpPath != "":
            self.questionIds = list(self.refExperience.keys())
   
        self.eval_questionIds = []
        if eval_question_info_file != "":
            for qId in self.eval_question_info.keys():
                convId = qId.split("-")[0] + "-" + qId.split("-")[1]
                if not convId in self.eval_questionIds:
                    self.eval_questionIds.append(convId)
        else:
            self.eval_questionIds = self.questionIds

        #initialize action and state space (states can have structured or unstructured representation)
        action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=14, name='action')
        if structured:
            observation_spec = array_spec.ArraySpec(shape=(4,), dtype=np.float32, name='observation')
        else:
            observation_spec = array_spec.ArraySpec(shape=(768,), dtype=np.float32, name='observation')
            observation_spec2 =  {'observation': array_spec.ArraySpec(shape=(768,), dtype=np.float32, name='observation') ,
            'structure': array_spec.ArraySpec(shape=(4,), dtype=np.float32, name='structure')}
        

        #path for loading/storing RCS model chekpoints
        self.check_num = check_num
        self.loadPath = loadPath
        self.storePath = storePath
        #path where augmented training data should be stored
        self.storeQuestionInfoPath = question_info_path

        #if actionmasking should be used: certain actions are masked out in observation
        if action_mask:
            splitter = self.observation_and_action_constraint_splitter
        else:
            splitter = None
        tf.compat.v1.enable_v2_behavior()
        train_step_counter = tf.compat.v2.Variable(0)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        initializer1 = tf.keras.initializers.GlorotUniform(seed=self.seed_value)
        initializer2 = tf.keras.initializers.GlorotUniform(seed=(self.seed_value+1))

        #define Q-network
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu, name="dense1",kernel_initializer=initializer1)
        self.dense2 = tf.keras.layers.Dense(15, name="dense2", kernel_initializer=initializer2)
        q_net = sequential.Sequential([self.dense1,self.dense2], input_spec=tensor_spec.from_spec(observation_spec))
     
        if self.action_mask:
            os = observation_spec2
        else:
            os = observation_spec

        #define DQN Agent
        self.ActionSamplerAgent = RCSDqnAgent(
            time_step_spec(tensor_spec.from_spec(os)), 
            tensor_spec.from_spec(action_spec), 
            q_network=q_net,
            optimizer=optimizer,
            observation_and_action_constraint_splitter=splitter,
            epsilon_greedy=None,
            boltzmann_temperature=0.3,
            gamma=1.0,
            #emit_log_probability=True,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
        self.checkpoint = tf.train.Checkpoint(q_network=q_net)
        self.ActionSamplerAgent.initialize()
        self.collect_policy = self.ActionSamplerAgent.collect_policy
        self.eval_policy = self.ActionSamplerAgent.policy

        self.rg_batch_size = batch_size
        self.final_obs = False
        self.ref_counter = 0
        self.current_epoch = 0

        #data structures for storing infos about which categories are chosen
        self.choiceCount = []
        self.secondChoiceCount = []
        combinedCount = dict()
        for i in range(15):
            self.choiceCount.append(0)
            self.secondChoiceCount.append(0)
        self.totalCount = 0
        self.secondtotalCount = 0
        self.missCount = 0
        

    def reset(self):
        self.final_obs = False
        #TODO: remove this in case we want to use original questions as well
        random.shuffle(self.questionIds)
        self.ref_counter = 0
        avg_rewardList = []

   
    def saveModel(self):
        #save checkpoints for each epoch
        self.checkpoint.save(self.storePath + "/ckpt") #config["checkpoint_path"] + "-seed-"+ str(config["seed"])
     

    def loadModel(self):
        self.checkpoint.restore(self.loadPath + "/ckpt-" + str(self.check_num)).expect_partial()


    def updateModel(self, traj, iter):
        avg_rewards = tf.reduce_mean(traj.reward).numpy()  
        train_loss = self.ActionSamplerAgent.train(traj)      
        print("AS iteration ", iter, ": avg_rewards = ", avg_rewards, ", loss = ", train_loss.loss.numpy(), flush=True)
        return 


    def cycle(self):
        iter = 0
        while not self.final_obs:   
            traj = self.collectExperienceBatchQLearning()
            if not traj is None:
                self.updateModel(traj, iter)
                iter +=1
        
        self.current_epoch += 1
        print("current epoch: ", self.current_epoch)
        return
    

    def getNextStructuredRep(self, current_structure, curr_act):
        curr_act = int(curr_act)
        if curr_act == 0:
            return current_structure
        elif curr_act == 1:
            return [1] + current_structure[1:]
        elif curr_act == 2:
            return [0] + current_structure[1:]
        elif curr_act == 3:
            return [current_structure[0]] + [1] + current_structure[2:]
        elif curr_act == 4:
            return [current_structure[0]] + [0] + current_structure[2:]
        elif curr_act == 5:
            return current_structure[:2] + [1] + [current_structure[-1]]
        elif curr_act == 6:
            return current_structure[:2] + [0] + [current_structure[-1]]
        elif curr_act == 7:
            return current_structure[:3] + [1]
        elif curr_act == 8:
            return current_structure[:3] + [0]
        elif curr_act == 9:
            return current_structure
        elif curr_act == 10:
            return current_structure
        elif curr_act == 11:
            return current_structure
        elif curr_act == 12:
            return current_structure
        elif curr_act == 13:
            return [0] + [current_structure[1]] + [1] + [current_structure[3]]
        elif curr_act == 14:
            return [0] + current_structure[1:]
        return current_structure
        

    def getObservations(self, convIds, nextStructuredRep=None):
        removeList = []
        encoded_questions = None
        for convId in convIds:
            origId = convId.split("-")[0] + "-" + convId.split("-")[1] + "-0-0"
            #get question encoding
            if self.structured:
                if not nextStructuredRep is None:
                    #note that here convid is in fact next question id collected for the second step
                    enc_quest = np.asarray(nextStructuredRep[convId])
                    enc_quest = tf.expand_dims(enc_quest, axis=0)
                else:
                    if not "structured_rep" in self.question_info[origId].keys():
                        print("no structured rep for question: ", origId)
                        removeList.append(convId)
                        continue
                    enc_quest = np.asarray(self.question_info[origId]["structured_rep"])
                    enc_quest = tf.expand_dims(enc_quest, axis=0)
      
            else:
                if not nextStructuredRep is None and convId in self.question_info.keys():
                    
                    enc_quest = self.question_info[convId]["encoded_question"]
                else:
                    enc_quest = self.question_info[origId]["encoded_question"]
          
            if encoded_questions is None:
                encoded_questions = enc_quest
            else:
                encoded_questions = tf.keras.layers.concatenate([encoded_questions,  enc_quest], axis=0)
            
            
       
        if encoded_questions is None:
            return None
        #observations = tf.expand_dims(encoded_questions, axis=1)
        observations = tf.dtypes.cast(encoded_questions, tf.float32)
       
        for rId in removeList:
            convIds.remove(rId)
        
        return observations
    
    def getObservations_actionMasking(self, convIds, nextStructuredRep=None):
        encoded_questions = None
        #encode observations consisting of encoded question and structured representation (used for action masking)
        for convId in convIds:
            origId = convId.split("-")[0] + "-" + convId.split("-")[1] + "-0-0"
            if not nextStructuredRep is None:
                quest_struct = np.asarray(nextStructuredRep[convId])
                enc_quest = self.question_info[convId]["encoded_question"]
            else:
                enc_quest = self.question_info[origId]["encoded_question"]
                if not "structured_rep" in self.question_info[origId].keys():
                    print("no structure available: ", origId)
                    quest_struct = np.asarray([0,1,0,0])
                else:
                    quest_struct = np.asarray(self.question_info[origId]["structured_rep"])
            quest_struct = tf.expand_dims(quest_struct, axis=0)
              
            if encoded_questions is None:
                encoded_questions = enc_quest
                structure = quest_struct
            else:
                encoded_questions = tf.keras.layers.concatenate([encoded_questions,  enc_quest], axis=0)
                structure = tf.keras.layers.concatenate([structure,  quest_struct], axis=0)
            
            
        if encoded_questions is None:
            return None
       # observations = tf.expand_dims(encoded_questions, axis=1)
        encoded_questions = tf.dtypes.cast(encoded_questions, tf.float32)
        structure = tf.dtypes.cast(structure, tf.float32)
        
        observations = {"observation": encoded_questions, "structure": structure} #, "available_refs": avail_refs}
        return observations


    def getNextObservations(self, convIds, actions):
        nextIds = []
        nextStructuredRep = dict()
        for i in range(len(convIds)):
            #get next question id based on action selected for current step
            curr_act = actions[i].numpy()
            nextId = convIds[i] + "-" + str(curr_act) + "-0"
            nextIds.append(nextId)
            if self.structured or self.action_mask:
                origId = convIds[i] + "-0-0"
                if not "structured_rep" in self.question_info[origId].keys():
                    print("no structure available: ", origId)
                    current_structure = np.asarray([0,1,0,0])
                else:
                    current_structure = self.question_info[origId]["structured_rep"]
                nextStructuredRep[nextId] = self.getNextStructuredRep(current_structure, curr_act)
        
        if self.action_mask:
            next_observations = self.getObservations_actionMasking(nextIds, nextStructuredRep)
        else:
            next_observations = self.getObservations(nextIds, nextStructuredRep)
        return next_observations
    


    def getRewards(self, questionList, actions, prevActions=None):
        rewards = []
       
        for i in range(len(questionList)):
            qId = questionList[i]
            convId = qId.split("-")[0] + "-" + qId.split("-")[1]
            if not convId in self.refExperience.keys():
                print("warning: no experience collected for: ", convId)
                rewards.append(-1.0)
                continue
            act = str(actions[i].numpy())
            ref_cat_id = act
            #in case of multistep: new id consists of concatenation of previous and current action
            if not prevActions is None:
                prev_act =str(prevActions[i].numpy())
                ref_cat_id= prev_act + "-" + act
            if ref_cat_id in self.refExperience[convId].keys():    
                # if isinstance(self.refExperience[convId][ref_cat_id], dict):
                #     refNum = random.sample(list(self.refExperience[convId][ref_cat_id].keys()),1)[0]
                #     rewards.append(self.refExperience[convId][ref_cat_id][refNum])
                # else:
                rewards.append(self.refExperience[convId][ref_cat_id])
            else:
                rewards.append(-1.0)      
          

        avg_rewardList.extend(rewards)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
       
        return rewards


    def collectExperienceBatchQLearning(self):
        if self.ref_counter + self.rg_batch_size > len(self.questionIds)-1:
            print("end of training samples: empty observation returned")
            self.final_obs = True
            return None
    
        #get next training conversation ids
        convIds = self.questionIds[self.ref_counter:self.ref_counter+self.rg_batch_size]
       
        self.ref_counter+= self.rg_batch_size
        #encode observation
        if self.action_mask:
            observations = self.getObservations_actionMasking(convIds)
        else:
            observations = self.getObservations(convIds)
        time_step = ts.restart(observations)
        #take an action
        action_step = self.collect_policy.action(time_step, seed=self.seed_value)
        actions = action_step.action  
        #get rewards
        rewards = self.getRewards(convIds, actions)
        #get next observation
        next_observations = self.getNextObservations(convIds, actions)
        #store the collected experience as one trajectory
        if self.action_mask:
            first_questions = observations["observation"]
            second_questions = next_observations["observation"]
            first_questions = tf.expand_dims(first_questions, axis=1)
            second_questions = tf.expand_dims(second_questions, axis=1)
            encoded_questions = tf.concat((first_questions, second_questions), axis=1)
            first_structure = observations["structure"]
            second_structure = next_observations["structure"]
            first_structure = tf.expand_dims(first_structure, axis=1)
            second_structure = tf.expand_dims(second_structure, axis=1)
            structure = tf.concat((first_structure, second_structure), axis=1)
            observations = {"observation": encoded_questions, "structure": structure}
        else:
            observations = tf.expand_dims(observations, axis=1)
            next_observations = tf.expand_dims(next_observations, axis=1)
            observations = tf.concat((observations, next_observations), axis=1)
        rewards = tf.expand_dims(rewards, axis=1)
        rewards = tf.concat((rewards, rewards), axis=1)
        batch_size = len(actions)
        #next_observations = tf.expand_dims(next_observations, axis=1)
        if not self.multi_step:
            discounts = tf.zeros([batch_size,2], dtype=tf.float32)
            step_type = tf.zeros([batch_size,1], dtype=tf.int32)
            next_step_type =  tf.repeat(2, [batch_size])
            next_step_type = tf.expand_dims(next_step_type, axis=1)
            new_step_type = tf.concat((step_type, next_step_type), axis=1)
            next_step_type = tf.concat((next_step_type,step_type), axis=1)
            actions = tf.expand_dims(actions, axis=1)
            actions = tf.concat((actions, actions), axis=1)
            traj = Trajectory(new_step_type, observations, actions, (), next_step_type, rewards, discounts)
        else:    
            next_time_step = ts.transition(next_observations, rewards)
            next_action_step = self.collect_policy.action(next_time_step, seed=self.seed_value)
            next_actions =next_action_step.action
            next_rewards = self.getRewards(convIds, next_actions, actions)

            first_step_type = tf.zeros([batch_size,1], dtype=tf.int32)
            mid_step_type =  tf.repeat(1, [batch_size])
            last_step_type =  tf.repeat(2, [batch_size])
            mid_step_type = tf.expand_dims(mid_step_type, axis=1)
            last_step_type = tf.expand_dims(last_step_type, axis=1)
            step_type = tf.concat((first_step_type, mid_step_type), axis=1)
            next_step_type = tf.concat((mid_step_type,last_step_type), axis=1)
            actions = tf.expand_dims(actions, axis=1)
            next_actions = tf.expand_dims(next_actions, axis=1)
            actions = tf.concat((actions, next_actions), axis=1)
            discount = tf.repeat(1.0, [batch_size])
            discount = tf.expand_dims(discount, axis=1)
            discount = tf.concat((discount,discount), axis=1)
            
            #second step:
            second_encoded_questions = tf.concat((second_questions, second_questions), axis=1)
            second_structures = tf.concat((second_structure, second_structure), axis=1)
            next_rewards = tf.expand_dims(next_rewards, axis=1)
            second_rewards = tf.concat((next_rewards, next_rewards), axis=1)
            second_discount = tf.zeros([batch_size,2], dtype=tf.float32)
            second_step_type = tf.concat((mid_step_type, last_step_type), axis=1)
            second_next_step_type = tf.concat((last_step_type,last_step_type), axis=1)
            second_actions = tf.concat((next_actions, next_actions), axis=1)

            step_type = tf.concat((step_type, second_step_type), axis=0)
            next_step_type = tf.concat((next_step_type, second_next_step_type), axis=0)
            discount = tf.concat((discount, second_discount), axis=0)
            combined_encoded_questions = tf.concat((encoded_questions, second_encoded_questions), axis=0)
            combined_structures = tf.concat((structure, second_structures), axis=0)
            combined_observations = {"observation": combined_encoded_questions, "structure": combined_structures}
            #observations = tf.concat((observations, second_observations), axis=0)
            rewards = tf.concat((rewards, second_rewards), axis=0)
            actions = tf.concat((actions, second_actions), axis=0)
       
            traj = Trajectory(step_type, combined_observations, actions, (), next_step_type, rewards, discount)
       
       
        return traj


    def doEvalStep_actionMasking(self, qId, secondStep=False):
        origId = qId.split("-")[0] + "-" + qId.split("-")[1] + "-0-0"
        if not "structured_rep" in self.eval_question_info[origId].keys():
            current_structure =[0,1,0,0]
            print("Warning: no struct rep available for eval question with id: ", qId)
        else:
            current_structure = self.eval_question_info[origId]["structured_rep"]
        if not secondStep:
            quest_struct = np.asarray(current_structure)
        else:
            curr_act = qId.split("-")[2]
            quest_struct = np.asarray(self.getNextStructuredRep(current_structure, curr_act))
        
        if qId in self.eval_question_info.keys():
            enc_quest = self.eval_question_info[qId]["encoded_question"]
        else:
            print("Warning: no reformulation for selected action, refid: ", qId)
            enc_quest = tf.zeros([1,768])
      
        quest_struct = tf.expand_dims(quest_struct, axis=0)
        enc_quest = tf.dtypes.cast(enc_quest, tf.float32)
        quest_struct = tf.dtypes.cast(quest_struct, tf.float32)
        observations = {"observation": enc_quest, "structure": quest_struct}
        time_step = ts.restart(observations)
        
        action_step = self.eval_policy.action(time_step, seed=self.seed_value)
        distribution = self.eval_policy.get_distribution()
    
        all_actions = np.arange(15)
        all_actions = tf.expand_dims(all_actions, axis=1)
        log_probability_scores = distribution.log_prob(all_actions)
        log_probability_scores = tf.transpose(log_probability_scores)
        return log_probability_scores, distribution


    def doEvalStep(self, qId, secondStep=False):

        if self.structured:
            if not secondStep:
                enc_quest = np.asarray(self.eval_question_info[qId]["structured_rep"])
            else:
                origId = qId.split("-")[0] + "-" + qId.split("-")[1] + "-0-0"
                curr_act = qId.split("-")[2]
                current_structure = self.eval_question_info[origId]["structured_rep"]
                enc_quest = np.asarray(self.getNextStructuredRep(current_structure, curr_act))
        else:
            if qId in self.eval_question_info.keys():
                enc_quest = self.eval_question_info[qId]["encoded_question"]
            else:
                print("Warning: no reformulation for selected action, refid: ", qId)
                enc_quest = tf.zeros([1,768])

        observations = tf.dtypes.cast(enc_quest, tf.float32)
        observations = tf.expand_dims(observations, axis=0)
        time_step = ts.restart(observations)
        
        action_step = self.eval_policy.action(time_step, seed=self.seed_value)
        distribution = self.eval_policy.get_distribution()
        all_actions = np.arange(15)
        all_actions = tf.expand_dims(all_actions, axis=1)
        log_probability_scores = distribution.log_prob(all_actions)
        log_probability_scores = tf.transpose(log_probability_scores)
        return log_probability_scores, distribution
    

    def selectActions(self, convId, topQuestions,  log_probability_scores, distribution, prevAction=None):
        origId = convId + "-0-0"
        if self.actionSelection == "greedy":
            topActionsProbs, topActions = tf.math.top_k(log_probability_scores,self.sampleSize)
        elif self.actionSelection == "sampling":
            topActions =  tf.nest.map_structure(lambda d: d.sample(self.sampleSize, seed=self.seed_value),distribution)
            topActions = tf.transpose(topActions)
            #if not self.structured:
             #   topActions = topActions[0]
       
        firstTopIds = []
        print("top act: ", topActions)
        for i in range(self.sampleSize):
            if prevAction: 
                if int(topActions[0][i].numpy()) == 0:
                    topqId = convId + "-" + str(prevAction) + "-0" 
                elif int(prevAction) == 0:
                    topqId = convId + "-" + str(topActions[0][i].numpy()) + "-0" 
                else:
                    topqId = convId + "-" + str(prevAction) + "-0-" + str(topActions[0][i].numpy()) 
            else:
            #TODO: change -0 at end if we want to use several refs for same question and category
                topqId = convId + "-" + str(topActions[0][i].numpy()) + "-0"
            firstTopIds.append(topqId)
            
            if topqId in self.eval_question_info.keys():
                if prevAction:
                    self.secondChoiceCount[int(topActions[0][i].numpy())] +=1
                    self.secondtotalCount +=1
                else:
                    self.choiceCount[int(topActions[0][i].numpy())] +=1
                    self.totalCount +=1
                c = 0
                actualTopqId = topqId
                while topqId in topQuestions.keys():
                    c+=1
                    #print("topqid already contained!! ", topqId)
                    topqId = topqId[:-1] + str(c)
                topQuestions[topqId] = self.eval_question_info[actualTopqId]
                topQuestions[topqId]["answers"] = self.eval_question_info[origId]["answers"]
            else:
                self.missCount += 1
        return topActions, firstTopIds


    def createEvalBatch(self):
        topQuestions = dict()
        for convId in self.eval_questionIds:
            origId = convId + "-0-0"
            if self.structured:
                if not "structured_rep" in self.eval_question_info[origId].keys():
                    print("no structured rep for question: ", origId)
                    topQuestions[origId] = self.eval_question_info[origId]
                    continue
                else:
                    print("struct rep: ", self.eval_question_info[origId]["structured_rep"])

            if self.addOriginalQuestions:
                topQuestions[origId] = self.eval_question_info[origId]

            if self.action_mask:
                log_probability_scores, distribution = self.doEvalStep_actionMasking(origId, False)
            else:
                log_probability_scores, distribution = self.doEvalStep(origId, False)
            topActions, topqIds = self.selectActions(convId, topQuestions,  log_probability_scores, distribution)
            if self.multi_step: 
                for i in range(len(topActions)):
                    if self.action_mask:
                        next_log_probability_scores, next_distribution = self.doEvalStep_actionMasking(topqIds[i], True)
                    else:
                        next_log_probability_scores, next_distribution = self.doEvalStep(topqIds[i], True)
                    self.selectActions(convId, topQuestions,  next_log_probability_scores, next_distribution, topActions[0][i].numpy())
                         
        for i in range(15):
            self.choiceCount[i] = self.choiceCount[i]/self.totalCount
            if self.multi_step:
                self.secondChoiceCount[i] = self.secondChoiceCount[i]/self.secondtotalCount
          
      
        print("selected category distribution first step: ", self.choiceCount)
        print("total number of selected categories: ", self.totalCount)
        print("number of times selected category was not available: ", self.missCount)
        with open(self.storeQuestionInfoPath , "wb") as outfile:
            pickle.dump(topQuestions, outfile)
        
        return 


    def isActionAllowed(self,struct_rep, refCat):
        possible = True
        if struct_rep[0] == 0:
            
            if refCat in [2,10,13,14]:
                possible = False
        else:
            if refCat == 1:
                possible = False             
        if struct_rep[1] == 0:
            if refCat in [4,9]:
                possible = False
        else:
            if refCat == 3:
                possible = False
        if struct_rep[2] == 0:
            if refCat in [6,11]:
                possible = False
        else:
            if refCat == 5:
                possible = False
        if struct_rep[3] == 0:
            if refCat in [8,12]:
                possible = False
        else:
            if refCat == 7:
                possible = False

        return possible


    def observation_and_action_constraint_splitter(self, observations):
        #get observation and mask separately - derive mask based on structured rep
        mask = np.ones([observations["observation"].shape[0],15], dtype=np.int32)
        j = 0
        for struct_rep in observations["structure"].numpy():
            for i in range(15):
                if not self.isActionAllowed(struct_rep, i):
                    mask[j][i] = 0
            j += 1
          
        return observations["observation"], mask


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--storepath", dest="store_path")
    parser.add_argument("--loadpath", dest="load_path", default="")
    parser.add_argument("--reffile", dest="ref_file", default="")
    parser.add_argument("--questioninfo", dest="question_info", default="")
    parser.add_argument("--evalquestioninfo", dest="eval_question_info", default="")
    parser.add_argument("--storequestioninfo", dest="store_question_info")
    parser.add_argument("--structured", action='store_true')
    parser.add_argument("--actionmask", action='store_true')
    parser.add_argument("--actionselection", dest="action_selection", default="greedy")
    parser.add_argument("--samplesize", dest="sample_size", type=int, default=1)
    parser.add_argument("--checknum", dest="checknum", type=int, default=1)
    parser.add_argument("--learningrate", dest="learn_rate", type=float, default=1e-5)
    parser.add_argument("--batchsize", dest="batch_size", type=int, default=10)
    parser.add_argument("--epochs", dest="epochs", type=int, default=1)
    parser.add_argument("--multistep", action='store_true')
    parser.add_argument("--evalonly", action='store_true')
    parser.add_argument("--addoriginalquestions", action='store_true')


    args = parser.parse_args()
    RG = RCSModelBase(args.question_info, args.eval_question_info,  args.ref_file,args.load_path, args.store_path, args.checknum, args.learn_rate, args.batch_size, args.store_question_info, args.structured, args.actionmask, args.action_selection, args.sample_size, args.multistep, args.addoriginalquestions)
    if args.load_path != "":
        print("model loaded")
        RG.loadModel()
  
    i = 0
    print("eval only mode: ", args.evalonly, flush=True)
    if not args.evalonly:
        while i < int(args.epochs):
            RG.reset()
            RG.cycle()
            i += 1
            finalAvgReward = np.mean(avg_rewardList)
            print("AVG RG rewards in cycle ", i , ": ", finalAvgReward)
        RG.saveModel()
   
    RG.createEvalBatch()
    
