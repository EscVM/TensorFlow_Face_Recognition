#--------------------------
#Date: 19/12/2019
#Place: Turin, PIC4SeR
#Author: Fra, Vitto
#Project: faceAssistant
#---------------------------

import os.path, time, datetime
from sys import exit
import pickle, json
import numpy as np
import cv2
from tqdm import tqdm
from textwrap import dedent
from faceNet.MyMQTT import MyMQTT
from faceNet.MyCam import MyCam
from faceNet.Timer import Timer
from faceNet.faceNet import faceNet
from sklearn.neighbors import KNeighborsClassifier


class CameraAcquistion(object):
    def __init__(self, conf_file="conf.json"):
        self.read_configuration(conf_file)
        # create a folder for the dataset if it doesn't exist
        if not os.path.exists(self.dataset):
            os.mkdir('dataset')
        self.cam = MyCam(self.cameras_file,self.default_camera)  
        self.frame = None
        self.model = faceNet(self.bb_model,self.emb_model)
        self.every_frame = 5
    
    
    def read_configuration(self,conf_file):
        conf = json.loads(open(conf_file,'r').read())
        self.ROOT_DIR = os.path.abspath('')
        
        self.logo = open(conf["logo_file"]).read()
        self.dataset = conf["dataset"]
        self.data_file = conf["database_file"]
        self.cameras_file = conf["cameras_file"]
        self.default_camera = conf["default_camera_index"]
        
        self.bb_model = conf["bb_model"]
        self.emb_model = conf["emb_model"]
        self.classifier_model_path = conf["classifier_model"]
        self.nframes = conf["acquisition_nframes"]
    
    
    
    def menu(self):
        print(dedent("""Do you want to add a person?
                         - Y
                         - N
                    """))
        make = input('> ')

        if make.lower() == 'y':
            print(dedent("""What is the name?"""))
            name = input('> ')
            print(dedent("""What is the surname?"""))
            surname = input('> ')
            folder_name = name.lower() + '_' + surname.lower()

            self.target_folder = os.path.join(self.dataset, folder_name)

            if not os.path.exists(self.target_folder):
                os.mkdir(self.target_folder)

            self.start_acquisition()
        else:
            cv2.destroyAllWindows()
            self.cam.release()
            print(dedent("""Do you want to create embeddings?
                        - Y
                        - N
                        """))
            make = input("> ")

            if make.lower() == 'y':
                self.createEmbeddings()
                self.stop()
            else:
                self.stop()


    def start_acquisition(self):
        cv2.namedWindow('Camera Acquisition', cv2.WINDOW_NORMAL)
        # create switch for ON/OFF functionality
        switch = 'REC : OFF \nREC : ON'
        cv2.createTrackbar(switch, 'Camera Acquisition',0,1, self.info_recording)

        while True:
            rec = cv2.getTrackbarPos(switch, 'Camera Acquisition')

            if rec == 0:
                self.frame = self.cam.read()[1]

                # drawing ellipse
                cv2.ellipse(self.frame, (self.frame.shape[1] // 2,self.frame.shape[0] // 2),
                            (130,200),0,0,360,(192,192,192), 1) 
                cv2.imshow('Camera Acquisition', self.frame)
                
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    self.stop()
            else:
                index = 0
                try:
                    for i in tqdm(range(self.nframes)):
                        frames = []
                        thresh = []
                        for j in range(self.every_frame):
                            self.frame = self.cam.read()[1]
                            frames.append(self.frame.copy())
                            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                            thresh.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                            # adding text
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(self.frame, 'Recording', (20, 100), font, 1, (0,0,255), 2, cv2.LINE_AA)
                            cv2.ellipse(self.frame, (self.frame.shape[1] // 2,self.frame.shape[0] // 2),
                                        (130,200),0,0,360,(192,192,192), 1)     
                            cv2.imshow('Camera Acquisition', self.frame)
                            k = cv2.waitKey(1) & 0xFF
                        frame_to_save = frames[thresh.index(max(thresh))]
                        path = os.path.join(self.target_folder, 'img_{}.png'.format(index))
                        index += 1
                        cv2.imwrite(path, frame_to_save)
                        if k == 27:
                            self.stop()

                    cv2.destroyAllWindows()
                    self.menu()
                    
                except Exception as e:
                    print(str(e))
                    pass

            

    def stop(self):
        cv2.destroyAllWindows()
        self.cam.release()
        print("\n[INFO] Bye!")
        exit(0)

        
        
    def info_recording(self, x):
        print("[INFO] Start Recording")


   
    def random_colors(self,N):
        np.random.seed(1)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
        for i in colors:
            yield i

    
    
    def createEmbeddings(self):
        # grab the paths to the input images in our dataset
        print("[INFO] Processing the dataset with the Embeddings Network\n\n")
        
        database = {}
        people = [x for x in sorted(os.listdir(self.dataset)) if x[0]!="."] #exclude hidden folders
        colors = self.random_colors(len(people))
        
        # loop over the folders
        for name in people:
            # initialize the dictionary for the name
            new_person = {"color": next(colors),"encodings":[]}
            subpath = self.dataset + "/" + name + "/"
            for img in tqdm(sorted(os.listdir(subpath)),desc=name):
                # load the input image and convert it from BGR (OpenCV ordering) to RGB
                try:
                    image = cv2.imread(subpath + img)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print("[WARNING] Found a non-image file in", name, "folder:", img)
                    
                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input image
                if np.max(rgb.shape) > 1000:
                    minsize = 100
                else:
                    minsize = 75
                    
                boxes,_ = self.model.get_face_locations(rgb,minsize)

                if not boxes.size:
                    print("[WARNING] Face not found for ", name,img)
                    continue

                # compute the facial embedding for the face
                faces = self.model.get_faces(rgb, boxes)
                encodings = self.model.get_embeddings(faces)

         
                #add the encodings to the database
                #(we suppose a single face per image)
                new_person["encodings"].append(encodings[0])
            database[name]= new_person
        
        # save database to file
        print("\n[INFO] Saving the embeddings.")
        f = open(self.data_file, "wb+")
        f.write(pickle.dumps(database))
        f.close()
        
        #knn classifier
        print("[INFO] Training knn classifier.")
        X = [] 
        y = []
        for name in database:
            X += database[name]["encodings"]
            y += [name]*len(database[name]["encodings"])
        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, y)
        print("[INFO] Saving the knn classifier.")
        f = open(self.classifier_model_path, "wb+")
        f.write(pickle.dumps(neigh))
        f.close()
            


if __name__ == '__main__':
    main = CameraAcquistion()
    main.menu()