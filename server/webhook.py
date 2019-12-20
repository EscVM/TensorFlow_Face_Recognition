from flask import Flask, request, make_response, jsonify
import time, json, datetime
from MyMQTT import MyMQTT

######################################################################

#initialize the flask app
app = Flask(__name__)


#deafult route
@app.route('/')
def index():
    return "This is the Virtual Security Asssistant server."


#create a route for webhook
@app.route('/webhook', methods = ['POST'])
def webhook():
    #return response
    return make_response(jsonify(results()))


#function for responses
def results():
    #build a request object
    req = request.get_json(force=True)
    
    #fetch action from json
    intent = req.get('queryResult').get('intent').get('displayName')
    lang = req.get('queryResult').get('languageCode')
    
    #check action
    if intent == "face_recognition":
        try:
            date = req.get('queryResult').get('parameters').get('date')
            speech_text,display_text = read_seen_file(date) #generate the texts
        except Exception as e:
            speech_text = "Error in reading 'seen.json' file."
            display_text = "Error in reading 'seen.json' file.\n" + str(e)
    elif intent == "change_camera":
        n = int(req.get('queryResult').get('parameters').get('number'))
        try:
            speech_text,display_text = change_camera(n) #generate the texts
        except Exception as e:
            speech_text = "Error in changing the camera."
            display_text = "Error in changing the camera.\n" + str(e)
    elif intent == "now":
        try:
            speech_text,display_text = now()
        except Exception as e:
            speech_text = "Error in reading 'seen.json' file."
            display_text = "Error in reading 'seen.json' file.\n" + str(e)
    else:
        return {'fullfillmentText':"Intent not implemented."}
    #return a fullfillment response
    resp = {"payload": {"google": {"expectUserResponse": True,"richResponse":{"items":[{"simpleResponse": {
            "displayText": display_text,
            "textToSpeech": speech_text
    }}]}}}}
    return resp


#list the recognized people
def read_seen_file(date):
    seen = json.loads(open(conf['seen_file'], 'r').read())
    
    date = get_date(date)

    if date is None:
    	text = "La data richiesta non è disponible."
    	return text,text

    #today&yesterday
    if (date.today() - date).days == 0:
        d = "Today "
    elif (date.today() - date).days == 1:
        d = "Yesterday "
    else:
        d = date.strftime("%A %d %B ")
    
    date_key = date.strftime("%Y %m %d")
    if date_key not in seen["list"]:
        text = d + "I didn't see anyone."
        return text,text
    
    speech_text =  d + "I saw:\n"
    display_text = speech_text

    for person in seen["list"][date_key]:
        if person["name"] == "Unknown":
            continue
        tm = time.localtime(person["time"])
        speech_text += person["name"] + " at " + str(tm.tm_hour) + " " + str(tm.tm_min) + ","
        display_text += person["name"] + " at " + "{:02d}".format(tm.tm_hour) + "." + "{:02d}".format(tm.tm_min) + "\n"        
    return speech_text,display_text 


#parse the Dialogflow date and check if it is whithin the past one week range
def get_date(date):
    if date == "":
        return datetime.date.today()
    
    #parse date
    date = date[:10].split('-')
    date = datetime.date(int(date[0]),int(date[1]),int(date[2]))
    
    #check if date is allowed (-1 week,+1 week)
    delta = (date - date.today()).days
    if delta > 6 or delta < -6:
        return None #date out of range
    
    if delta >= 0:
        return date
    else: #convert next week to past week(due to Dialogflow date interpretation)
        return date - datetime.timedelta(days=7)


#list who has been recognized in the past 5 seconds
def now():
    seen = json.loads(open(conf['seen_file'],'r').read())
    found = False
    text = "Now I see:\n"
    date_key = time.strftime("%Y %m %d")
    for person in seen["list"][date_key]:
        if time.time()-person["time"] <= 5:
            if person["name"]!="Unknown":
                text += person["name"] + '\n'
                found = True
    if not found:
        text = "I don't see anyone." 
        
    
#change camera
def change_camera(n):
    cameras = json.loads(open(conf['camera_file'], 'r').read())
    allowed_n = [x["index"] for x in cameras] 
    
    if n not in allowed_n:
        text = "Camera with index " + str(n) + " not available."
        return text,text

    #create the message  
    mess = {"camera" : n, "time" : int(time.time())}
    mess = json.dumps(mess)
    #send new camera info over MQTT  
    MQTTclient.publish(MQTT_topic["camera"],mess)
    text = "Camera with index " + str(n) + " selected." 
    return text,text



#run the app
if __name__ == '__main__':
    #read configuration file
    conf = json.loads(open('conf.json').read())
    #start MQTT clinet (publisher for camera number)
    MQTTclient = MyMQTT(conf['MQTT_ID'], conf['MQTT_broker'], conf['MQTT_user'], conf['MQTT_pwd'])
    MQTTclient.start()
    #wait for connnection
    while not MQTTclient.is_connected:
        time.sleep(0.1)
    app.run(host='0.0.0.0')