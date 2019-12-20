#--------------------------
#Date: 19/12/2019
#Place: Turin, PIC4SeR
#Author: Fra, Vitto
#Project: faceAssistant
#---------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid annoying logs

import cv2
import numpy as np
import tensorflow as tf


class profileNet():  
    def __init__(self, model = "bin/profile/mobile.pb", verbose=True):             
        if verbose:
            print("[ProfileNet] Creating tf session.")
        # Create session and load graph
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.tf_sess = tf.compat.v1.Session(config=tf_config)
        
        # import graph
        self.importGraph(model,verbose)
        
        if verbose:
            print("[ProfileNet] Done.\n")



    def importGraph(self, model, verbose=False):
        output_names = ['dense_1/Sigmoid']
        input_names = ['input_2']

        def get_frozen_graph(graph_file):
            """Read Frozen Graph file from disk."""
            with tf.io.gfile.GFile(graph_file, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            return graph_def
        
        if verbose:
            print("[ProfileNet] Reading frozen graph '{}'.".format(model))
            
        trt_graph = get_frozen_graph(model)
        
        if verbose:
            print("[ProfileNet] Importing the graph.")

        tf.import_graph_def(trt_graph, name='')

        # input and output tensor names.
        self.input_tensor_name = input_names[0] + ":0"
        output_tensor_name = output_names[0] + ":0"

        self.output_tensor = self.tf_sess.graph.get_tensor_by_name(output_tensor_name)        
        

        
    def normalize(self,imgs):
        for i,img in enumerate(imgs):
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),(224,224))
            img = (np.array(img)/255)[...,None]
            img = np.concatenate((img,img,img),axis=-1)
            imgs[i] = img
        return np.array(imgs)
        
        
    
    def predict(self,imgs,thr):
        if imgs.shape[1:] != (224,224,3):
            raise ValueError("Wrong image shape. Expected (n,224,224,3), got " + str(imgs.shape))

        feed_dict = {
            self.input_tensor_name: imgs
        }

        pred = self.tf_sess.run(self.output_tensor, feed_dict)[...,0]
        debug = False
        if debug:
            for i in range(len(pred)):
                cv2.imwrite("stuff/img_{}.png".format(pred[i]),imgs[i]*255)
        return [True if p<thr else False for p in pred]
