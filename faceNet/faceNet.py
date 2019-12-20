#--------------------------
#Date: 19/12/2019
#Place: Turin, PIC4SeR
#Author: Fra, Vitto
#Project: faceAssistant
#---------------------------

##################################################################
# Networks adapted from https://github.com/davidsandberg/facenet #
##################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid annoying logs

import tensorflow as tf
import time
import faceNet.detectFace as detectFace
import numpy as np
import cv2

class faceNet():  
    def __init__(self,model_bb='bin/bb/frozen_graph',model_emb='bin/emb/frozen_graph',conf_thr=0.7,
                 fps_bb=False, fps_emb=False,verbose=True):
        self.graph_bb = tf.Graph()
        self.graph_emb = tf.Graph()
        self.import_graph(model_bb,model_emb,verbose)
        self.sess_bb = tf.compat.v1.Session(graph=self.graph_bb)
        self.sess_emb = tf.compat.v1.Session(graph=self.graph_emb)
        self.conf_thr = conf_thr
        self.fps_bb = fps_bb
        self.fps_emb = fps_emb
    
    
    
    def import_graph(self,model_bb,model_emb,verbose):
        if verbose:
            print("[faceNet] Importing bounding boxes graph.")
        with self.graph_bb.as_default():
            with tf.io.gfile.GFile(model_bb,'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        if verbose:
            print("[faceNet] Importing embeddings graph.")
        with self.graph_emb.as_default():
            with tf.io.gfile.GFile(model_emb,'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        if verbose:
            print("[faceNet] Done.\n")
                
                
    
    def get_face_locations(self,frame,minsize=20,factor=0.709,
                           threshold = [ 0.6, 0.7, 0.7 ]):
        if self.fps_bb:
            start = time.time()
        boxes, landmarks = detectFace.detect_face(frame, minsize,
                                                   threshold,factor,self.sess_bb)
        if self.fps_bb:
            delta = time.time()-start
            print('[faceNet] Bounding boxes: time:',delta,'fsp:',1/delta)
            
        return boxes, np.transpose(landmarks)
     
        
        
    def get_embeddings(self,faces):     
        faces = np.array([detectFace.prewhiten(cv2.resize(face,(160,160))) for face in faces])
        if self.fps_emb:
            start = time.time()                
        emb = self.sess_emb.run('embeddings:0',
                       feed_dict={'input:0': faces,
                                  'phase_train:0': False})
        if self.fps_emb:
            delta = time.time()-start
            print('[faceNet] Embeddings: time:',delta,'fsp:',1/delta)
        return emb
        
        
        
    def get_faces(self,frame,boxes,margin=60):
        faces = []
        for (left, top, right, bottom, conf) in boxes:
            left = np.maximum(left-margin/2,0)
            top = np.maximum(top-margin/2,0)
            right = np.minimum(right+margin/2,frame.shape[1])
            bottom = np.minimum(bottom+margin/2,frame.shape[0])
            top = int(top); right = int(right);
            bottom = int(bottom); left = int(left)
            face = frame[top:bottom,left:right]
            faces.append(face)
        return faces
        
        
        
    def compare_faces(self,storage_emb,emb,distance_thr=0.6,verbose=False):
        dist = np.sqrt(np.sum(np.square(np.subtract(storage_emb, emb)),axis=-1))
        if verbose:
            print("[faceNet] Distances:",dist)
        return dist < distance_thr
