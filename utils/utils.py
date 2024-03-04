import json
import re


with open("kb/dicts/labels.json", "r") as labelFile:
    labels_dict = json.load(labelFile)

def hasNumber(string):
    for ch in string:
        if ch.isdigit():
            return True
    return False

# return if the given string is a timestamp
def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    try:
        if not(pattern.match(timestamp)):
            return False
        else:
            return True
    except Exception as e:
        print("error for timestamp check: ", timestamp)
        return False

def convertTimestamp( timestamp):
    yearPattern = re.compile('^[0-9][0-9][0-9][0-9]-00-00T00:00:00Z')
    monthPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-00T00:00:00Z')
    dayPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    timesplits = timestamp.split("-")
    year = timesplits[0]
    if yearPattern.match(timestamp):
        return year
    month = convertMonth(timesplits[1])
    if monthPattern.match(timestamp):
        return month + " " + year
    elif dayPattern.match(timestamp):
        day = timesplits[2].rsplit("T")[0]
        return day + " " + month + " " +year
   
    return timestamp


# convert the given month to a number
def convertMonth( month):
    return{
        "01": "january",
        "02": "february",
        "03": "march",
        "04": "april",
        "05": "may",
        "06": "june",
        "07": "july",
        "08": "august",
        "09": "september", 
        "10": "october",
        "11": "november",
        "12": "december"
    }[month]

def getLabel(entity):
    label = ""
    if entity.startswith("Q") or entity.startswith("P"):
            #for predicates: P10-23, split away counting
        if "-" in entity:
            e = entity.split("-")[0]
        else:
            e = entity
        if e in labels_dict.keys():
            if not labels_dict[e] is None and len(labels_dict[e]) > 0:
                label = labels_dict[e][0]
           
    else:
        if is_timestamp(entity):
            label = convertTimestamp(entity)
        elif entity.startswith("+"):
            label = entity.split("+")[1]
        else:
            label = entity
    #print("label: ", label)
    return label

def getActionLabel(path):
    action_labels = []

    #for a in paths:
    p_labels = ""
    #use this if startpoint should be included in action:
 
    p_labels = getLabel(path[0]) + " "
    for aId in path[1]:
        p_labels += getLabel(aId) + " "
    #use this if endpoint should be included in action 
    p_labels += getLabel(path[2])
    action_labels.append(p_labels)
    return action_labels

