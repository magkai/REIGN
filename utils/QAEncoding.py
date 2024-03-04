import json
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
import tensorflow as tf

from transformers import BertTokenizer, TFBertModel
sys.path.append("utils")
import utils as ut
import time
import requests
import tagme

tagme.GCUBE_TOKEN = ""



class QAEncoder():

    def __init__(self, tagmeThreshold=0.3):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = TFBertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.TAGME_THRESHOLD = tagmeThreshold
        self.sparql_endpoint_url = "https://query.wikidata.org/sparql"



    ######ENCODING########################################

    def getWd_ids(self,title):
        success = False
        while not success:
            try:
                entity_id = requests.get('https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&languages=en&sites=enwiki&titles=' + title).json()
                success = True
            except:
                print("wd id exception")
                sys.stdout.flush()
                time.sleep(5)
                
        #print("entity id: ", entity_id)
        for key in entity_id.get("entities").keys():
            return key
        return ""

    def getQuestionEntities(self,question):
        tagged_entities = []       
        annotations = tagme.annotate(question)
        for ann in annotations.get_annotations(self.TAGME_THRESHOLD):
            wd_id = self.getWd_ids(ann.entity_title)
            if not wd_id == "":
                newEntry = {"id": wd_id, "label":ut.getLabel(wd_id), "mention": ann.mention}
                found = False
                for entry in tagged_entities:
                    if newEntry["id"] == entry["id"]:
                        found = True
                if not found:
                    tagged_entities.append(newEntry)
            #  tagged_entities.append({"title": ann.entity_title, "mention": ann.mention, "link_probability": ann.score, "wd_id": wd_id})
        return tagged_entities

    def getAliasesSparql(self, pId):
        query = '''SELECT ?wdAltLabel
                {
                VALUES (?wdt) {(wdt:''' + pId + ''')}
                ?wd wikibase:directClaim ?wdt .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                }'''
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        # TODO adjust user agent; see https://w.wiki/CX6
        sparql = SPARQLWrapper(self.sparql_endpoint_url, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()

   
    #get pre-trained BERT embeddings for question
    def encodeQuestions(self, questions):
        tokenized_input =  self.tokenizer(questions, return_tensors="tf", padding=True) 
        encoded_input = self.encoder(tokenized_input, output_hidden_states=True)
        #take average over all hidden layers
        all_layers = [ encoded_input.hidden_states[l] for l in range(1,13)]
        encoder_layer = tf.concat(all_layers, 1)
        pooled_output = tf.reduce_mean(encoder_layer, axis=1)

        return pooled_output



    #get pre-trained BERT embeddings for actions
    def encodeActions(self, actions):
        try:
            tokenized_actions =  self.tokenizer(actions, return_tensors="tf", padding=True, truncation=True, max_length=50) 
        except Exception as e:
            print("error: ", e) 
            print("actions not working: ", actions)
            return None
    
        encoded_actions = self.encoder(tokenized_actions, output_hidden_states=True)
        #take average overall all hidden layers
        all_layers = [ encoded_actions.hidden_states[l] for l in range(1,13)]
        encoder_layer = tf.concat(all_layers, 1)
        pooled_output = tf.reduce_mean(encoder_layer, axis=1)
    
        return pooled_output


    #get node labels for each path (this can be adapted to also include start (paths) and endpoint as action)
    def getActionLabels(self, paths):
        action_labels = []
    
        for a in paths:
            if a is None or None in a:
                print("None in path: ", a)
                continue
            #p_labels = ""
            #use this if startpoint should be included in action:
            p_labels = ut.getLabel(a[0]) + " "
            for aId in a[1]:
                p_labels += ut.getLabel(aId) + " "
            #use this if endpoint should be included in action 
            p_labels += ut.getLabel(a[2])
            action_labels.append(p_labels)
        return action_labels

    def getPredicateLabels(self, paths):
        action_labels = []
    
        for a in paths:
            if a is None or None in a:
                print("None in path: ", a)
                continue
            #p_labels = ""
            #use this if startpoint should be included in action:
            p_labels = "" # ut.getLabel(a[0]) + " "
            for aId in a[1]:
                p_labels += ut.getLabel(aId) + " "
            #use this if endpoint should be included in action 
            #p_labels += ut.getLabel(a[2])
            action_labels.append(p_labels)
        return action_labels

    #get node labels for each path (this can be adapted to also include start (paths) and endpoint as action)
    def getActionLabelsClocq(self, paths):
        action_labels = []
    
        for path in paths:
            p_labels = ""
            for p in path:
                #p_labels = ""
                #use this if startpoint should be included in action:
                p_labels += ut.getLabel(p) + " "
            action_labels.append(p_labels)
            #print("p labels : ", p_labels)
        return action_labels

    def getJointQuestionActionEncodings(self, action_labels, question):
        action_nbrs = 0
        #for start in action_labels.keys():
            #store how many paths are available per startpoint
        action_nbrs = len(action_labels)
        if action_nbrs == 0:
            return [], 0
        for k in range(len(action_labels)):
            action_labels[k] = question + " " + action_labels[k]
        first = True
        encoded_paths = None
        j = -1
    
        #encode paths batchwise
        for i in range(action_nbrs):   
            j+=1
            if j == 64:
                if first:
                    encoded_paths = self.encodeActions(action_labels[i-j:i+1])
                    if encoded_paths is None:
                        j = -1
                        continue
                    first = False
                else:
                    encoded_actions = self.encodeActions(action_labels[i-j:i+1])
                    if encoded_actions is None:
                        j = -1
                        continue
                    encoded_paths = tf.keras.layers.concatenate([encoded_paths, encoded_actions],axis=0)
                j = -1
        encoded_actions = self.encodeActions(action_labels[i-j:i+1])
        if encoded_actions is None and encoded_paths is None:
            return [],0
        if not encoded_actions is None:
            if first:
                encoded_paths = encoded_actions
            else:
                encoded_paths = tf.keras.layers.concatenate([encoded_paths, encoded_actions],axis=0)
        #pad all paths to length of 1000
        if len(encoded_paths) < 1000:
            zeros = tf.zeros((1000-action_nbrs, 768))
            encoded_paths = tf.keras.layers.concatenate([encoded_paths, zeros],axis=0)
        
        return encoded_paths, action_nbrs


    #get all action embeddings for paths in the dataset
    def getActionEncodings(self, action_labels):
        action_nbrs = 0
        #for start in action_labels.keys():
            #store how many paths are available per startpoint
        action_nbrs = len(action_labels)
        if action_nbrs == 0:
            return [], 0
        first = True
        encoded_paths = None
        j = -1
        #encode paths batchwise
        for i in range(action_nbrs):   
            j+=1
            if j == 64:
                if first:
                    encoded_paths = self.encodeActions(action_labels[i-j:i+1])
                    if encoded_paths is None:
                        j = -1
                        continue
                    first = False
                else:
                    encoded_actions = self.encodeActions(action_labels[i-j:i+1])
                    if encoded_actions is None:
                        j = -1
                        continue
                    encoded_paths = tf.keras.layers.concatenate([encoded_paths, encoded_actions],axis=0)
                j = -1
        encoded_actions = self.encodeActions(action_labels[i-j:i+1])
        if encoded_actions is None and encoded_paths is None:
            return [],0
        if not encoded_actions is None:
            if first:
                encoded_paths = encoded_actions
            else:
                encoded_paths = tf.keras.layers.concatenate([encoded_paths, encoded_actions],axis=0)
        #pad all paths to length of 1000
        if len(encoded_paths) < 1000:
            zeros = tf.zeros((1000-action_nbrs, 768))
            encoded_paths = tf.keras.layers.concatenate([encoded_paths, zeros],axis=0)
        
        return encoded_paths, action_nbrs