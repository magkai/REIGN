import json
from weakref import ReferenceType
import numpy as np
from transformers import TFBartForConditionalGeneration, BartTokenizer, TFTrainer, TFTrainingArguments
import tensorflow as tf
import json 
import pickle
import os
import random
import sys
from argparse import ArgumentParser
import copy

random.seed(7)

BART_CACHE_PATH = "models/"
MAX_LENGTH = 50
SAMPLES_PER_CAT = 2000

CAT_TOKENS = ["rc1", "rc2","rc3","rc4","rc5","rc6", "rc7","rc8","rc9", "rc10", "rc11", "rc12", "rc13", "rc14"]


class ReformulationGenerator(tf.Module):
    def __init__(self):
        super(ReformulationGenerator, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir=BART_CACHE_PATH)
     

  
    def train(self, train_dataset, dev_dataset):
     
        trainer = TFTrainer(
            model=self.model, args=self.training_args, train_dataset=dev_dataset, eval_dataset=train_dataset
        )
        trainer.train()


    def createTrainer(self, storepath, runname, stepsize, epochs):
        print("epochs: ", epochs, "stepsize: ", stepsize)
        self.training_args = TFTrainingArguments(
            output_dir=storepath,  # output directory
            num_train_epochs=int(epochs),  # total number of training epochs
            learning_rate=3e-5,
            per_device_train_batch_size=10,  # batch size per device during training
            per_device_eval_batch_size=10,  # batch size for evaluation
            #warmup_steps=200,  # number of warmup steps for learning rate scheduler
           # weight_decay=0.01,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            logging_strategy="epoch",
            save_strategy="steps",
           # evaluation_strategy="steps",
            do_train="True",
         #   do_eval="True",
            run_name=runname,
            eval_steps = 2455,
            #logging_steps=1000,
            save_steps=stepsize#3220#4076#1635#4371#683
        )
        with self.training_args.strategy.scope():
            self.model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-base", cache_dir=BART_CACHE_PATH)
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir=BART_CACHE_PATH)
        


    def createDatasets(self, train_inputs, train_outputs, dev_inputs, dev_outputs):
        
        train_input_encodings = self.tokenizer(train_inputs, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="tf")
        train_output_encodings = self.tokenizer(train_outputs, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="tf")

        dev_input_encodings = self.tokenizer(dev_inputs, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="tf")
        dev_output_encodings = self.tokenizer(dev_outputs, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="tf")

        # create dataset objects
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_input_encodings),
            train_output_encodings["input_ids"]
        ))
       
        dev_dataset = tf.data.Dataset.from_tensor_slices((
            dict(dev_input_encodings),
            dev_output_encodings["input_ids"]
        ))
       
        return train_dataset, dev_dataset

    

    def prepareDataWithCategories(self,data_info, all_data_info, catString=""):
        
        if catString == "":
            categories = list(range(1, 15))
        else:
            categories = catString.split(",")
        inputs = []
        outputs = []
        catCount =dict()
        for i in range(15):
            catCount[i] = 0
        for qId in data_info.keys():

            if qId.count("-")> 3:
                quest_id = qId.split("-")[0] + "-" + qId.split("-")[1] + "-" +  qId.split("-")[2] +"-0"
            else:
                quest_id =  qId.split("-")[0] + "-" + qId.split("-")[1] +"-0-0"
            

            if qId == quest_id:
                continue
            inQuestion = all_data_info[quest_id]["question"]
            inHistory = all_data_info[quest_id]["conv_history"]
            inCategory = data_info[qId]["ref_category"]
            
            if not str(inCategory) in categories:
                continue
           
            CAT_TOK = CAT_TOKENS[inCategory-1]
        
            catCount[inCategory] += 1
            outQuestion = data_info[qId]["question"]
            if qId == quest_id:
                continue
            inputs.append(inHistory + " " + inQuestion + " " + CAT_TOK)
            
            outputs.append(outQuestion)

        print("inputs: " , inputs[0:30])
        print("outputs: ", outputs[0:30])
        print("number of total samples: ", len(inputs))
        print("number per cat: ", catCount)
        return (inputs, outputs)


    def prepareData(self, data_info, catString=""):
    
        if catString == "":
            categories = []
        else:
            categories = catString.split(",")
        print("categories: ", categories)
        inputs = []
        outputs = []
        
        for qId in data_info.keys():
            quest_id = qId.split("-")[0] + "-" + qId.split("-")[1] + "-0-0"
            if qId == quest_id:
                continue
            inQuestion = data_info[quest_id]["question"]
            inHistory = data_info[quest_id]["conv_history"]
            outQuestion = data_info[qId]["question"]
            inCategory = data_info[qId]["ref_category"]
            if not str(inCategory) in categories:
                continue
    
            if inHistory == "":
                inputs.append(inQuestion) 
            else:    
                inputs.append(inHistory + " " + inQuestion)
            
            outputs.append(outQuestion)

        print("inputs: " , inputs[0:30])
        print("outputs: ", outputs[0:30])
        print("number of total samples: ", len(inputs))
        return (inputs, outputs)


    def shuffleData(self, inputs, outputs):
        zipped = list(zip(inputs, outputs))
        random.shuffle(zipped)
        shuffled_inputs, shuffled_outputs  =  zip(*zipped)
        return (shuffled_inputs, shuffled_outputs)


    def selectData(self, data):
        new_data = dict()
        cat_data = dict()
        for i in range(1,15):
            cat_data[i]  = []
        for qId in data.keys():
            if qId.count("-")> 5:
                print("qid skip: ", qId)
                continue
            if data[qId]["ref_category"] == 0:
                new_data[qId] = copy.deepcopy(data[qId])
                continue
            cat_data[data[qId]["ref_category"]].append(qId)

        for cat in cat_data.keys():
            cat_ids = random.sample(cat_data[cat], SAMPLES_PER_CAT)
            for cid in cat_ids:
                new_data[cid] = copy.deepcopy(data[cid])
        
        ref_cat_num = []
        for i in range(15):
            ref_cat_num.append(-1)
        for qId in new_data.keys():
            if len(qId.split("-")) > 4:
                refcat = qId.split("-")[4]
            else:
                refcat = qId.split("-")[2]
            ref_cat_num[int(refcat)] += 1

        print("ref cats: ", ref_cat_num, flush=True) 
        return new_data


if __name__ == "__main__":
    parser = ArgumentParser()
 
    parser.add_argument("--refoption", dest="ref_option", default="")
    parser.add_argument("--refcategories", dest="ref_categories", default="")
    parser.add_argument("--epochs", dest="epochs", type=int, default=3)
    parser.add_argument("--storepath", dest="store_path")
    parser.add_argument("--runname", dest="run_name")
    parser.add_argument("--devdata", dest="dev_data")
    parser.add_argument("--traindata", dest="train_data")
    args = parser.parse_args()
    
    RefGen = ReformulationGenerator()
    #train BART model with categories as input
    if args.ref_option == "onepercategory":
        with open(args.train_data , "rb") as infile:
            train_data_info = pickle.load(infile)
        train_data = RefGen.selectData(train_data_info)
        train_inputs, train_outputs = RefGen.prepareDataWithCategories(train_data, train_data_info, args.ref_categories)
        train_inputs, train_outputs = RefGen.shuffleData(train_inputs, train_outputs)

        with open(args.dev_data , "rb") as infile:
            dev_data_info = pickle.load(infile)
        dev_data = RefGen.selectData(dev_data_info)
        dev_inputs, dev_outputs = RefGen.prepareDataWithCategories(dev_data, dev_data_info, args.ref_categories)
        dev_inputs, dev_outputs = RefGen.shuffleData(dev_inputs, dev_outputs)
    #train BART model without categories
    else:
        train_inputs, train_outputs = RefGen.prepareData(args.train_data, args.ref_categories)
        train_inputs, train_outputs = RefGen.shuffleData(train_inputs, train_outputs)

        dev_inputs, dev_outputs = RefGen.prepareData(args.dev_data, args.ref_categories)
        dev_inputs, dev_outputs = RefGen.shuffleData(dev_inputs, dev_outputs)
    
    train_dataset, dev_dataset = RefGen.createDatasets(train_inputs, train_outputs, dev_inputs, dev_outputs)
    
    step_size = int(len(dev_inputs)/10)
    print("step size: ", step_size, flush=True)
    RefGen.createTrainer(args.store_path, args.run_name, step_size, args.epochs)
    #in our case use devset for training and trainset for eval
    RefGen.train(train_dataset, dev_dataset)
    print("Done with Training", flush=True)
