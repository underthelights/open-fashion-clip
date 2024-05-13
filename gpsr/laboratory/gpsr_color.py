import rospy
import openai
import json
import re
import warnings

# Read data from file
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# Parse object.md
def parse_objects(data):
    parsed_objects = re.findall(r'\|\s*(\w+)\s*\|', data, re.DOTALL)
    parsed_objects = [objects for objects in parsed_objects if objects != 'Objectname']
    parsed_objects = [objects.replace("_", " ") for objects in parsed_objects]
    parsed_objects = [objects.strip() for objects in parsed_objects]

    parsed_categories = re.findall(r'# Class \s*([\w,\s, \(,\)]+)\s*', data, re.DOTALL)
    parsed_categories = [category.strip() for category in parsed_categories]
    parsed_categories = [category.replace('(', '').replace(')', '').split() for category in parsed_categories]
    parsed_categories_plural = [category[0] for category in parsed_categories]
    parsed_categories_plural = [category.replace("_", " ") for category in parsed_categories_plural]
    parsed_categories_singular = [category[1] for category in parsed_categories]
    parsed_categories_singular = [category.replace("_", " ") for category in parsed_categories_singular]

    if parsed_objects or parsed_categories:
        return parsed_objects, parsed_categories_plural, parsed_categories_singular
    else:
        warnings.warn("List of objects or object categories is empty. Check content of object markdown file")
        return []

# Make Category to Object Dictionary
def extractCategory2obj(markdown_content):
    category_pattern = re.compile(r'\# Class (\w+) \((\w+)\)')
    object_pattern = re.compile(r'\| (\w+)  \|')

    objects_dict = {}
    categorySing2Plur = {}
    categoryPlur2Sing = {}
    current_category_plur = None
    current_category = None

    for line in markdown_content.split('\n'):
        category_match = category_pattern.match(line)
        if category_match:
            current_category_plur = category_match.group(1)
            current_category = category_match.group(2)

            categorySing2Plur[current_category] = current_category_plur
            categoryPlur2Sing[current_category_plur] = current_category

            objects_dict[current_category] = []

        object_match = object_pattern.match(line)
        if object_match and current_category:
            object_name = object_match.group(1)
            objects_dict[current_category].append(object_name)

    return objects_dict, categoryPlur2Sing, categorySing2Plur

objects_file_path = 'task/gpsr_repo/object.md'
objects_data = read_data(objects_file_path)
object_names, object_categories_plural, object_categories_singular = parse_objects(objects_data)
category2objDict, categoryPlur2Sing, categorySing2Plur = extractCategory2obj(objects_data)

def followup(cmd):
    print(cmd)

### HELP Functions ###
def get_yolo_bbox(agent, category=None):
    yolo_bbox = agent.yolo_module.yolo_bbox

    if category:
        print(category)
        print(category2objDict)
        categoryItems = category2objDict[category]
        yolo_bbox = [obj for obj in yolo_bbox if agent.yolo_module.find_name_by_id(obj[4]) in categoryItems]

    while len(yolo_bbox) == 0:
        print("No objects detected")
        rospy.sleep(1)
        yolo_bbox = agent.yolo_module.yolo_bbox

    return yolo_bbox

def move_gpsr(agent, loc):
    agent.move_abs(loc)
    rospy.sleep(2)  
    print(f"[MOVE] HSR moved to {loc}")

def pick(obj):

    print(f"[PICK] {obj} is picked up")

def place(obj, loc):
    # [TODO] Implement how the object can be placed at the location
    print(f"[PLACE] {obj} is placed at {loc}")

### HRI and People Perception Commands ###

color_list = ["blue", "yellow", "black", "white", "red", "orange", "gray"]
clothe_list = ["t shirt", "shirt", "blouse", "sweater", "coat", "jacket"]


# "guideClothPrsFromBeacToBeac": "{guideVerb} the person wearing a {colorClothe} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}",
def guideClothPrsFromBeacToBeac(agent, params):
    # Take the person wearing a white shirt from the entrance to the trashbin
    # Escort the person wearing a yellow shirt from the pantry to the kitchen
    # Lead the person wearing a yellow t shirt from the kitchen table to the sofa
    # Escort the person wearing a blue sweater from the kitchen table to the office
    # Take the person wearing a gray sweater from   the storage rack to the living room
    params = {'guideVerb': 'Take', 'colorClothe': 'white shirt', 'fromLocPrep': 'from', 'loc': 'entrance', 'toLocPrep': 'to', 'loc_room': 'trashbin'}

    # [0] Extract parameters
    guide, color, loc, room = params['guideVerb'], params['colorClothe'], params['loc'], params['loc_room']

    # [1] Move to the specified location
    move_gpsr(agent, loc)

    # [2] Find the person in the location
    print(f"[FIND] the person wearing a {color} in the {loc}")

    # [3] Make the person to follow HSR to the room
    # follow


# "talkInfoToGestPrsInRoom": "{talkVerb} {talk} {talkPrep} the {gestPers} {inLocPrep} the {room}",
def greetClothDscInRm(agent, params):
    # Salute the person wearing a blue t shirt in the living room and follow them to the kitchen
    # Introduce yourself to the person wearing an orange coat in the bedroom and answer a quiz
    # Greet the person wearing a blue t shirt in the bedroom and answer a question
    # Introduce yourself to the person wearing a gray t shirt in the kitchen and say something about yourself
    # params = {'greetVerb': 'Salute', 'art': 'a', 'colorClothe': 'blue t shirt', 'inLocPrep': 'in', 'room': 'living room', 'followup': 'follow them to the kitchen'}

    # [0] Extract parameters
    greet, art, color, room, cmd = params['greetVerb'], params['art'], params['colorClothe'], params['room'], params['followup']

    # [1] Move to the specified room
    move_gpsr(agent, room)

    # [2] Find the person in the room
    print(f"[FIND] the person wearing {art} {color} in the {room}")

    # [3] Generate the followup command
    followup(cmd)

# "countClothPrsInRoom": "{countVerb} people {inLocPrep} the {room} are wearing {colorClothes}",
def countClothPrsInRoom(agent, params):
    # Tell me how many people in the kitchen are wearing red jackets
    # Tell me how many people in the living room are wearing black jackets
    # Tell me how many people in the bathroom are wearing white jackets
    
    # [0] Extract parameters
    room, color = params['room'], params['colorClothes']

    # [1] Move to the specified room
    move_gpsr(agent, room)

    # [2] Check the number of people wearing the specified color
    # [TODO] Color Detection, Clothes Detection
    count = 0
    print(f"[COUNT] {count} people in the {room} are wearing {color}")

    # params = {countVerb: 'Tell me how many', room: 'kitchen', colorClothes: 'red jackets'}

### TODO NOW ###
# "countObjOnPlcmt": "{countVerb} {plurCat} there are {onLocPrep} the {plcmtLoc}",
# Tell me how many drinks there are on the sofa
# Tell me how many drinks there are on the sofa
# Tell me how many cleaning supplies there are on the bedside table
# Tell me how many cleaning supplies there are on the shelf
# Tell me how many snacks there are on the tv stand
# Tell me how many dishes there are on the kitchen table

verbType2verb = {
    "{takeVerb}": ["take", "get", "grasp", "fetch"],
    "{placeVerb}": ["put", "place"],
    "{deliverVerb}": ["bring", "give", "deliver"],
    "{bringVerb}": ["bring", "give"],
    "{goVerb}": ["go", "navigate"],
    "{findVerb}": ["find", "locate", "look for"],
    "{talkVerb}": ["tell", "say"],
    "{answerVerb}": ["answer"],
    "{meetVerb}": ["meet"],
    "{tellVerb}": ["tell"],
    "{greetVerb}": ["greet", "salute", "say hello to", "introduce yourself to"],
    "{countVerb}": ["tell me how many"],
    "{followVerb}": ["follow"],
    "{guideVerb}": ["guide", "escort", "take", "lead"],
    # "{rememberVerb}": ["meet", "contact", "get to know", "get acquainted with"],
    # "{describeVerb}": ["tell me how", "describe"],
    # "{offerVerb}": ["offer"],
    # "{accompanyVerb}": ["accompany"]
}

verbType2cmdName = {
    "{takeVerb}": ["takeObjFromPlcmt"],
    "{placeVerb}": [],
    "{deliverVerb}": [],
    "{bringVerb}": ["bringMeObjFromPlcmt"],
    "{goVerb}": ["goToLoc"],
    "{findVerb}": ["findPrsInRoom", "findObjInRoom"],
    "{talkVerb}": ["talkInfoToGestPrsInRoom"],
    "{answerVerb}": ["answerToGestPrsInRoom"],
    "{meetVerb}": ["meetPrsAtBeac", "meetNameAtLocThenFindInRm"],
    "{tellVerb}": ["tellPrsInfoInLoc", "tellObjPropOnPlcmt", "tellCatPropOnPlcmt"],
    "{greetVerb}": ["greetClothDscInRm", "greetNameInRm"],
    "{countVerb}": ["countObjOnPlcmt", "countPrsInRoom", "countClothPrsInRoom"],
    "{followVerb}": ["followNameFromBeacToRoom", "followPrsAtLoc"],
    "{guideVerb}": ["guideNameFromBeacToBeac", "guidePrsFromBeacToBeac", "guideClothPrsFromBeacToBeac"],
    # "remember": [],
    # "describe": [],
    # "offer": [],
    # "accompany": []
}

cmdName2cmdStr = {
    "goToLoc": "{goVerb} {toLocPrep} the {loc_room} then {followup}",
    "takeObjFromPlcmt": "{takeVerb} {art} {obj_singCat} {fromLocPrep} the {plcmtLoc} and {followup}",
    "findPrsInRoom": "{findVerb} a {gestPers_posePers} {inLocPrep} the {room} and {followup}",
    "findObjInRoom": "{findVerb} {art} {obj_singCat} {inLocPrep} the {room} then {followup}",
    "meetPrsAtBeac": "{meetVerb} {name} {inLocPrep} the {room} and {followup}",
    "countObjOnPlcmt": "{countVerb} {plurCat} there are {onLocPrep} the {plcmtLoc}",
    "countPrsInRoom": "{countVerb} {gestPersPlur_posePersPlur} are {inLocPrep} the {room}",
    "tellPrsInfoInLoc": "{tellVerb} me the {persInfo} of the person {inRoom_atLoc}",
    "tellObjPropOnPlcmt": "{tellVerb} me what is the {objComp} object {onLocPrep} the {plcmtLoc}",
    "talkInfoToGestPrsInRoom": "{talkVerb} {talk} {talkPrep} the {gestPers} {inLocPrep} the {room}",
    "answerToGestPrsInRoom": "{answerVerb} the {question} {ofPrsPrep} the {gestPers} {inLocPrep} the {room}",
    "followNameFromBeacToRoom": "{followVerb} {name} {fromLocPrep} the {loc} {toLocPrep} the {room}",
    "guideNameFromBeacToBeac": "{guideVerb} {name} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}",
    "guidePrsFromBeacToBeac": "{guideVerb} the {gestPers_posePers} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}",
    "guideClothPrsFromBeacToBeac": "{guideVerb} the person wearing a {colorClothe} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}",
    "bringMeObjFromPlcmt": "{bringVerb} me {art} {obj} {fromLocPrep} the {plcmtLoc}",
    "tellCatPropOnPlcmt": "{tellVerb} me what is the {objComp} {singCat} {onLocPrep} the {plcmtLoc}",
    "greetClothDscInRm": "{greetVerb} the person wearing {art} {colorClothe} {inLocPrep} the {room} and {followup}",
    "greetNameInRm": "{greetVerb} {name} {inLocPrep} the {room} and {followup}",
    "meetNameAtLocThenFindInRm": "{meetVerb} {name} {atLocPrep} the {loc} then {findVerb} them {inLocPrep} the {room}",
    "countClothPrsInRoom": "{countVerb} people {inLocPrep} the {room} are wearing {colorClothes}",
    "tellPrsInfoAtLocToPrsAtLoc": "{tellVerb} the {persInfo} of the person {atLocPrep} the {loc} to the person {atLocPrep} the {loc2}",
    "followPrsAtLoc": "{followVerb} the {gestPers_posePers} {inRoom_atLoc}"
}

cmdName2cmdFunc = {
    "guideClothPrsFromBeacToBeac": guideClothPrsFromBeacToBeac,
    "greetClothDscInRm": greetClothDscInRm,
    "countClothPrsInRoom": countClothPrsInRoom,
}

# verbType2followUpStr
# followUpStr2followUp

class NoAppropriateVerbError(Exception):
   """
   This exception is raised when no appropriate verb is found for a given command.
   
   Attributes:
       command (str): The command for which no appropriate verb was found.
       message (str): An explanatory message about the exception.
   """

   def __init__(self, command, message="No appropriate verb found for the given command."):
       self.command = command
       self.message = message
       super().__init__(self.message)

   def __str__(self):
       return f"{self.message} Command: {self.command}"

# LOAD gpsr_config.json
def load_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
    return config

# CHAT w/ gpt-4
def chat(prompt):
    gpsr_config = load_config('gpsr_repo/gpsr_config.json')
    openai.api_key = gpsr_config['openai_api_key']
    model_engine = "gpt-4"

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].message.content

# ULTIMATE Text Parser
def ultimateParser(inputText):
    '''Ultimate parser for the inputText. It uses GPT-4 to parse the inputText.'''
    splitedInputText = inputText.split()
    mainVerb = splitedInputText[0]

    for verbType in verbType2verb:
        if mainVerb.lower() in verbType2verb[verbType]:
            candidateCmdStr = dict([cmdEntries for cmdEntries in cmdName2cmdStr.items() if cmdEntries[1].split()[0] == verbType])

            prompt = f'dict of {{cmdName: parameter}}: {candidateCmdStr}\n'
            prompt += f'inputText: {inputText}\n'
            prompt += 'return which cmdName the inputText is, and the parameters in ({})\n'
            prompt += 'you should only write with format: cmdName, {"parameterName": "parameterValue"}'
            
            gptAnswer = chat(prompt)

            splitIndex = gptAnswer.find(', ')
            cmdName = gptAnswer[:splitIndex]
            params = json.loads(gptAnswer[splitIndex+2:])

            ### TODO ###
            ### Catch Error and Retry

            print("[Parser] cmdName:", cmdName)
            print("[Parser] params:", params)

            return cmdName, params
            
    else: 
        raise NoAppropriateVerbError("No Appropriate Verb")
    
def nogadaParser(inputText):
    '''Handcrafted parser for the inputText'''
    ### TODO ###
    ### Make Handcrafted Parser
    pass

# MAIN
def gpsr(agent):

    # agent.say("I'm ready to receive a command")
    # rospy.sleep(4)

    # inputText, _ = agent.stt(10.)
    # cmdName, params = ultimateParser(inputText)

    # agent.say(f"Given Command is {cmdName}, Given Parameters {params}")
    # cmdFunc = cmdName2cmdFunc[cmdName]

    # cmdFunc(agent, params)

    tellCatPropOnPlcmt(agent, {})