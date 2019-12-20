#--------------------------
#Date: 19/12/2019
#Place: Turin, PIC4SeR
#Author: Fra, Vitto
#Project: faceAssistant
#---------------------------

import os, time, datetime
import pickle, json
import numpy as np
import cv2
from textwrap import dedent
from faceNet.MyMQTT import MyMQTT
from faceNet.MyCam import MyCam
from faceNet.Timer import Timer
from faceNet.profileNet import profileNet
from faceNet.faceNet import faceNet


class FaceNet(): 
    def __init__(self,conf_file="conf.json"):
        self.read_configuration(conf_file)
        print(self.logo)
        self.get_seen() # get seen dictionary from file
        self.MQTT_initialize()
        try:
            self.database = pickle.loads(open(self.data_file, "rb").read()) # read the storage database
            print("[INFO] Embeddings file imported.\n")
        except:
            raise FileNotFoundError("[Error] Encodings file not found. Generate it with 'imagesAcquisition.py'.")
        else: #knn model
            try:
                self.knn_model = pickle.loads(open(self.classifier_model_path, "rb").read()) # read the storage database
                print("[INFO] Knn classifier model imported.\n")
            except:
                raise FileNotFoundError("[Error] Knn classifier model not found. Generate it with 'imagesAcquisition.py'.")

        self.cam = MyCam(self.cameras_file,self.default_camera)       
        print("[INFO] Creating Tensorflow models...\n")
        self.profileNet = profileNet(self.profile_model)
        self.model = faceNet(self.bb_model,self.emb_model)
        self.counter = 0
    
    
    
    
    def run(self):
        self.del_old_thrd = Timer('deleteOld',"00.00",self.delete_old) # thread that deletes the old seen every day at midnight
        self.del_old_thrd.start()
        cv2.namedWindow('Camera', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.ready = True     
        self.frame = np.zeros((480,640,3),dtype='int8') #default frame
        while True:
            if self.ready:
                img = self.get_frame()
                boxes,names = self.detect(img)
                if boxes.size:
                    self.update_seen(names)
                img = self.draw_boxes(boxes,names)
                if not self.show(img,1):
                    break
            else:
                time.sleep(0.5)
        self.cam.release()
        quit()

    
    
    
    def read_configuration(self,conf_file):
        conf = json.loads(open(conf_file,'r').read())
        self.ROOT_DIR = os.path.abspath('')
        
        self.logo = open(conf["logo_file"]).read()
        self.data_file = conf["database_file"]
        self.seen_file = conf["seen_file"]
        self.cameras_file = conf["cameras_file"]
        self.default_camera = conf["default_camera_index"]
        
        self.bb_model = conf["bb_model"]
        self.emb_model = conf["emb_model"]
        self.profile_model = conf["profile_model"]
        self.classifier = conf["classifier"]
        if self.classifier:
            self.classifier_model_path = conf["classifier_model"]
        
        self.frame_width = conf["frame_max_width"]
        self.blur = conf["blur"]
        self.unknown_color = conf["unknown_color"]
        self.show_fps = conf["show_fps"]
        if self.show_fps:
            self.previous_prediction_time = time.time()
        
        self.line_width = conf["box_line_width"]
        self.font_dim = conf["font_dim"]
        
        self.MQTT_ID = conf["MQTT_ID"]
        self.MQTT_broker = conf["MQTT_broker"]
        self.MQTT_user = conf["MQTT_user"]
        self.MQTT_pwd = conf["MQTT_pwd"]
        self.MQTT_topic = conf["MQTT_topic"]


        

    def MQTT_initialize(self):
        self.MQTTclient = MyMQTT(self.MQTT_ID , self.MQTT_broker, self.MQTT_user, self.MQTT_pwd, self.dispatch)
        self.MQTTclient.start()
        #wait for connnection
        while not self.MQTTclient.is_connected:
            time.sleep(0.1)
        #subscribe
        for topic in self.MQTT_topic:
            self.MQTTclient.subscribe(topic)
        while sum(self.MQTTclient.is_subscribed) < len(self.MQTT_topic):
            time.sleep(0.1)

    
    
    
    def get_seen(self):
        try:
            self.seen = json.loads(open(self.seen_file, "r").read())
        except:
            print("[INFO] New seen.json file.\n")
            self.reset_seen()
            
    
    
    
    def reset_seen(self):
        self.seen = {"list":{}} # generate empty seen dataframe
        self.update_seen() # generate empty seen file    
    
    
    
        
    def delete_old(self):
        #max 7 days
        maxdate = datetime.date.today() - datetime.timedelta(days=7)

        seen = self.seen.copy()
        for t in seen["list"]:
            if datetime.strptime(t, "%Y %m %d") <= maxdate:
                del self.seen["list"][t]
                
    
    

    def dispatch(self,message):
        """
        Dispatch function for MQTT messages.
        :param message: MQTT message with JSON payload
        """
        topic = message.topic
        message = json.loads(message.payload.decode())
        
        print("[INFO] Received message on topic " + topic)
        
        if "camera" in topic:
            self.change_camera(message)
            
            


    def change_camera(self,message):
        """
        Change the camera.
        :param message: dictionary with 'camera' key -> camera ID to be selected, 't' key -> timestamp of the message
        """
        self.ready = False
        n = int(message.get('camera'))
        t = message.get('time')
        print("\n[INFO] " + time.ctime(t) + " Selected camera number " + str(n) + ".\n")
        self.cam.change(n)
        self.ready = True
        
        
    
    
    def get_frame(self):
        """
        Get the the image from the camera object. If the freame read fails it uses the previous frame. If it fails more than 10 times it tries to reattach the camera. It sets the self.frame attribute with the read image and returns the frame used for face localization and detection.
        :return: The frame.
        """
        frame = self.cam.read()
        
        if not frame[0]:
            if self.counter >= 10:
                self.cam.reattach()
                self.counter = -1
            frame = self.frame
            print('\n[INFO] Using previous frame.')
            self.counter += 1
        else:
            self.counter = 0
            frame = frame[1]
        
        if self.cam.flip: #usually we want to flip webcams horizontally
            frame = cv2.flip(frame, 1)
        
        self.frame = frame #frame that will be dispalyed
        if frame.shape[1]>self.frame_width:
            self.r = frame.shape[1] / self.frame_width
            frame = cv2.resize(frame, (self.frame_width,int(frame.shape[0]/self.r)), interpolation=cv2.INTER_AREA)
        else:
            self.r = 1

        return frame #frame used as tensorflow input
    
  

    
    def detect(self,frame=None):
        """
        Execute the detection and recognition algorithms with an image.
        :param frame: The image.
        :return: A tuple with the detected bounding boxes and the associated names.
        """
        if frame is None:
            return (None,None)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes,landmarks = self.model.get_face_locations(frame,self.cam.minsize,self.cam.factor,self.cam.thr)
        encodings = self.face_encodings(frame, boxes)
        names = []

        if self.classifier and np.argwhere(encodings).size != 0: # KNN model only if we have front faces
            # Check threshold to choose between known and unknown
            # only the front has to be considered
            print("Enc",encodings.shape)
            indexes = np.array([i for i in range(len(encodings)) if np.any(encodings[i])])
            print("Ind",indexes)
            
            closest_distances = self.knn_model.kneighbors(encodings[indexes], n_neighbors=1)[0]
            are_matches = [closest_distances[int(np.where(indexes==i)[0])][0] <= self.cam.distance_thr if i in indexes
                           else None for i in range(len(encodings))]
            print("Are_match",are_matches)
            # Predict classes and remove classifications that aren't within the threshold
            names = [name if rec else "Unknown" for name,rec in zip(self.knn_model.predict(encodings), are_matches)]
            return (boxes,names)

        # euclidean distances
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            name = 'Unknown' #default name
            if encoding is None:
                names.append(name)
                continue
            matches = {}
            
            #loop over the storage embeddings
            for db_name in self.database:
                person = self.database[db_name]
                person_match = self.model.compare_faces(person["encodings"], encoding,self.cam.distance_thr) #default is 0.6
                matches[db_name] = sum(person_match)
            
            #get name of maximum match
            if matches[max(matches,key=matches.get)]:
                name = max(matches,key=matches.get)
            # update the list of names
            names.append(' '.join([i.capitalize() for i in name.split('_')])) # write the name in a good way    
        return (boxes,names)
    
    
    
    
    def face_encodings(self,face_image, known_face_locations=None):
        """
        Given an image, return the 256-dimension face encoding for each face in the image. Function redefined from face_recognition to add side/front classification and use large model for facial landmarks.
        :param face_image: The image that contains one or more faces.
        :param known_face_locations: The bounding boxes of each face if you already know them.
        :return: A list of 256-dimensional face encodings (one for each face in the image).
        """
        if not known_face_locations.size:
            return np.array([])
        
        #if side faces: no landmarks and no encodings
        faces = self.model.get_faces(self.frame,known_face_locations*self.r)
        norm_faces = self.profileNet.normalize(faces.copy())
        are_front = self.profileNet.predict(norm_faces,self.cam.profile_thr)

        return np.array([self.model.get_embeddings(faces[i:i+1])[0] if are_front[i] else None
                for i in range(len(known_face_locations))])

    
    

    def update_seen(self,names=[]):
        date = datetime.date.today().strftime("%Y %m %d")
        if date not in self.seen["list"]:
            self.seen["list"][date] = []
        
        for name in names:
            seen_names_list = [d['name'] for d in self.seen["list"][date]]
            if name not in seen_names_list:
                new_seen = {"name":name,"time":time.time()}
                self.seen["list"][date].append(new_seen)
            else:
                index = seen_names_list.index(name)
                self.seen["list"][date][index]["time"] = time.time()
        
        # update seen file
        self.seen["updated"] = time.time()
        save_file = open(self.seen_file, "w+")
        json.dump(self.seen,save_file)
        save_file.close()
      
    
    
    
    def draw_boxes(self,boxes=[],names=[]):
        if self.frame is None:
            return None
        frame = self.frame.copy()
        
        if self.show_fps: # write fps
            fps = 1./(time.time() - self.previous_prediction_time)
            self.previous_prediction_time = time.time()
            text =  "FPS: {:.2f}".format(fps)
            cv2.putText(frame, text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, self.font_dim, self.unknown_color, 2) #top left corner
            
        # draw the faces boxes with names
        for ((left, top, right, bottom, conf), name) in zip(boxes, names):
            left = np.maximum(left,0)
            top = np.maximum(top,0)
            right = np.minimum(right,frame.shape[1])
            bottom = np.minimum(bottom,frame.shape[0])
            top = int(top*self.r); right = int(right*self.r); bottom = int(bottom*self.r); left = int(left*self.r)
         
            if not name == 'Unknown':
                color = self.database['_'.join(name.lower().split())]["color"]
            else:
                color = self.unknown_color
                if self.blur: #blur unknown faces
                    name = '' #no name
                    face_image = frame[top:bottom, left:right]
                    face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
                    frame[top:bottom, left:right] = face_image

            cv2.rectangle(frame, (left, top), (right, bottom), color, self.line_width)
            y = top - 15 if top - 15 > 15  else top + 20
            cv2.putText(frame, name, (left + 5, y), cv2.FONT_HERSHEY_SIMPLEX, self.font_dim, color, 2)

        return frame




    def show(self,frame,delay=0):
        """
        Visualized the modified frame.
        :param frame: The frame to be modified.
        :return: 0 if we want to stop the execution, 1 else
        """
        cv2.imshow("Camera", frame)
        k = cv2.waitKey(delay) & 0xFF
        if k == 27: 
            cv2.destroyAllWindows()
            return 0 # if the `esc` key was pressed, break from the loop
        elif k == ord('c'):
            self.cam.switch()      
        return 1



if __name__ == "__main__":
    model = FaceNet()
    model.run()
