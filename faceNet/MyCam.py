#--------------------------
#Date: 19/12/2019
#Place: Turin, PIC4SeR
#Author: Fra, Vitto
#Project: faceAssistant
#---------------------------

import cv2,json
import urllib.request
import getpass
import numpy as np

class MyCam():    
    def __init__(self, cameras_file='cameras.json', index=0):
        self.cameras_file = cameras_file
        try:
            self.cameras = json.loads(open(self.cameras_file).read())
        except:
            raise SystemError("No 'cameras.json' file.")
        self.method = None
        self.name = None
        self.index = None
        self.change(int(index))
        
        
    def change(self,index):
        self.changing = True
        if self.method == "CAP":
            self.release()
        
        found = False
        for cam in self.cameras:
            if cam["index"] == index:
                found = True
                break #cam is the right camera
        
        if not found:
            raise ValueError("Camera index not found.")

        self.method = cam['method']['type']
        
        if self.method == "CAP":
            self.cam = cv2.VideoCapture(cam['method']['add'])
            if not self.cam.isOpened():
                raise SystemError("Camera not detected with index " + str(index) +
                                  " at address " + str(cam['method']['add']) + ".")
        elif self.method == "URL":
            #usually ipcameras have passwords
            auth_user = cam['method']['usr']
            auth_passwd = cam['method']['pwd']
            passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            passman.add_password(None, cam['method']['add'], auth_user, auth_passwd)
            authhandler = urllib.request.HTTPBasicAuthHandler(passman)
            opener = urllib.request.build_opener(authhandler)         
            urllib.request.install_opener(opener)
            self.cam = cam['method']['add']
        else:
            raise ValueError("Wrong camera method.")
        
        self.name = cam["name"]
        self.index = cam["index"]
        self.distance_thr = cam["distance_thr"]
        self.profile_thr = cam["profile_thr"]
        self.minsize = cam["face_minsize"]
        self.factor = cam["bbNet_factor"]
        self.thr =  cam["bbNet_thr"]
        self.flip = cam["flip"]

        print("[INFO] Camera '" + str(self.name) + "' with index " + str(self.index) + " attached.\n")
        self.changing = False


        
        
    def switch(self):
        indexes = [cam['index'] for cam in self.cameras]
        
        new_index = indexes.index(self.index) + 1
        if new_index >= len(indexes):
            new_index = 0

        self.change(indexes[new_index])   
    
    
    
    
    def read(self):
        if self.method == "CAP":
            return self.cam.read()
        elif self.method == "URL":
            try:
                imgResp = urllib.request.urlopen(self.cam)
                imgNp = np.array(bytearray(imgResp.read()), dtype = np.uint8)
                frame = cv2.imdecode(imgNp, -1)
                return (True,frame)
            except:
                print("[WARNING] Cannot get image from " + self.name)
                return (False,None)
        else:
            raise ValueError("Camera not attached.")
              
    
    
    
    def release(self):
        if self.method == "CAP":
            self.cam.release()
        
        self.cam = None
        self.method = None
        
        #name&index remain the last attached
        print("[INFO] Camera released.\n")
    
    
    
    def reattach(self):
        self.change(self.index)

        
    
    def isOpened(self):
        if self.changing:
            return False
        try:
            if self.method == "CAP":
                return self.cam.isOpened()
            elif self.method == "URL":
                return True
        except:
            return False

        
        
    def update_list(self):
        try:
            self.cameras = json.loads(open(self.cameras_file).read())
        except:
            raise SystemError("No 'cameras.json' file.")