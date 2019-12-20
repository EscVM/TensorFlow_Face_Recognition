<h1 align="center"> ~ Virtual Security Assistant ~ </h1>

Have you ever desired a personal tireless [keeper](https://media.urbanpost.it/wp-content/uploads/2019/03/ScreenShot2016-01-19at4.18.36PM.jpg) able to recognize people passing through your gate and inform you whenever you want? Do you want to remove badges from your office and automatically and precise take note of all people coming in and out? Or, do you want to create your personal "[big brother](https://qph.fs.quoracdn.net/main-qimg-d30d05225145292c4e9186a4c8fac778)" world in your neighborhood?
If so, this repository is right for you. Powered by a cascade of four state of the art very stupid and completely unaware neural networks, this repository is giving you the possibility to recognize with a cheap and low range camera all persons transit in its field of view. Moreover, all detections data is saved and stored in a practical ".json" file that can be used for whatever you want. For example, we created a very simple server able to parse this file and interact with Google Assistant. So, when everything is set, you will be able to wake up your google assistant from whatever device you like and ask who have been spotted by your Virtual Security Assistant. 

All code have been tested on a NVDIA 2080 that can easilly give you more than 40 fps. Moreover, it perfectly works also on a [Jetson Xavier](https://www.nvidia.com/it-it/autonomous-machines/embedded-systems/jetson-agx-xavier/) that can be easily installed on the site where you want to deploy your sistem (let's move this AI on the edge).

**Side Notes**:
This repository is a stupid demo made in a hurry for our robotic center by me and [fsalv](https://github.com/fsalv). It has been a distraction from our research, but with the community help can really become an interesting and helpful project. We intentionally left a easy customizable framework in order to let the community work and improve this Virtual Security Assistant. We used [TensorFlow](https://www.tensorflow.org/) library and [TensorRT](https://developer.nvidia.com/tensorrt) to optimize graphs for embedded solutions. So, all networks run at FP16.

## How it works

Let's see very briefly how the general framework works. There are four networks running behind the curtains. They all worked in a cascade manner. So, the output of the first network is the starting point of the second one and so on. 

- **1. Faces detection:** the first one is a convolutional neural network (CNN) trained to search in an image the presence of faces. Its outputs are all possible bounding boxes related with detected faces. 
- **2. Facial landmarks detection:** for all detected faces (locations given by the bounding boxes detected by the previous network) we crop the portion of the face and we feed a second network to get [facial landmarks](https://miro.medium.com/max/828/1*AbEg31EgkbXSQehuNJBlWg.png). These are a bunch of points of detected faces that highlight certain specific locations like eye, nose, etc. We use these points to pose and project all faces more parallel as possible to the camera. Doing so, it increases the accuracy and the precision of the overall pipeline.
- **3. Face side detection:** before feeding the last network (4), with all detected faces (locations given by the bounding boxes detected by the first network), we feed a third CNN model. This network is responsible to detect if the face is in "side" position. This is a really important passage, because the rest of the pipeline doesn't work well with faces in side position. Indeed, in case of a side face, the projecting algorithm distorts so much the resulting face that the last network produces pretty random results. So, if a certain face is in side position we don't further process that particular face, but we only show a red bounding box around it (the framework gives you the possibility to blur this unrecognized face).
- **4. Embeddings generation:** this is the part I like most. Practically, for all detected not side projected face we use a final network to generate a vector (simply a list of numbers) of 256 elements. These are not random numbers, but are "attributes" that the network has learnt to give at every face is feeded with. As if we are ask to describe a face we start to say the color of the eyes, hair, etc...the same does the network, but with numbers. This is soo cool and it encapsulates most of the philosophy of Deep Learning. So, at the end of this long neural pipeline we have for each face, for example present in a video frame, a nice and clean vector that gives a very accurate representation of the corresponding face.

Let's see all together with a graphical representation:

![Flow_chart of the recognition proces](images/flow_chart_2.png)

First we use the first network to detect possible faces in the given image. Then, we find the facial landmarks of our detected favorite wizard. At the same time, the third network says us that we are good to go and so, we can safely feed the last one producing the so wanted embedded representation.

But, wait! Where is the recognition part? Hey, clever question!

After getting our embedded representations, we compare it with a database with known faces. Faces that we want to recognize with our Virtual Security Assistant. About this, the framework let you easily capture a video where then automatically extracts some frames that are processed by the already explained pipeline. At the end of this "video capturing" procedure we have for each person inserted in the system a list of N embedded vectors. So, simply at run time we compare our extracted new embeddings with our database list.

We can do that in three different ways: 

- with simply obtaining [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between our new ebedded and the database. This is simple and has pretty good results, but scales bad with many knwown persons inside the database. That because I have to perform a mesure for all persons and all their embeddings inside the database.
- second solution a solution is using k-neearest neighboorhood. The framework let you easily train a k-nn model with the acquired dataset and then perform inference with the trained model. This is faster and in most of the cases mantains a similar accuracy.
- finally, we can train a simple multi-layer perceptron trained to classify new embeddings. We didn't include this multi-layer perceptron feature in this repository for a series of unfortunate events.

Now, you should have all information to undestarstand what is going on in this repository. So, let's get started!


<p align="center">
  <img width="350" height="400" src="images/wishes.gif">
</p>

# 1.0 Getting Started
Python3 is required. We tested the project with TensorFlow 1.14, but it should work also with other versions and even TensorFlow 2.0. Keras is not required!
## 1.1 Installation
We used only a couple of libraries, but for your simplicity we also created a 'requirements.txt' file. We purposely didn't add TensorFlow installation in the txt file, because it's up to you to choose your prefered version (gpu, cpu, etc...). Moreover, TensorFlow installation can be sometime pretty easy as 'pip install', but usually can become a pain in the ****. Follow the official [guide](https://www.tensorflow.org/install) and good luck:)  

1. Clone this repository
   ```bash
   git clone https://github.com/EscVM/Virtual_Security_Assistant
   ```
2. Install the required packages
   ```bash
   pip3 install -r requirements.txt
   ```
If you want to install everything manually, these are the libraries we used:
- numpy
- opencv-python
- paho-mqtt
- scikit-learn
- tensorflow

# 2.0 Database embeddings creation
The first step is to create reference embeddings (vectors of 256 elements). As default option we acquire 50 frames (taking the best one every 5) from the main camera connected to the computer. Then, the code produces automatically all encodings of all persons acquired and train a simple [knn](https://it.wikipedia.org/wiki/K-nearest_neighbors) model with all generated encodings. Finally, the code saves encodings and knn model in two separate pickle files inside faceNet folder.

Nevertheless, you don't need to know all these things because it's a smart code (not more than its creators) and it will take care of all the process. Simply launch in a terminal, inside the project folder, the following command:

```bash
   python3 imagesAcquisition.py
   ```
# 3.0 Launch the Virtual Security Assistant
Once all subjects that you want to recognize are inserted in the database as explained in section [2.0](#2.0), it's time to launch the Virtual Security Assistant. Simply launch in a terminal, inside the project folder, the following command and the code will do all the heavy lifting :)

```bash
   python3 faceAssistant.py
   ```
# 4.0 Create a simple server fort Google Home Assistant

Coming soon...

# 5.0 Personalize the prject with the configuartions files

Coming soon...

# Citation
Use this bibtex if you want to cite this repository:
```
@misc{Virtual_Security_Assistant,
  title={Face recognition system with Google Home Assistant integration.},
  author={Vittorio, Francesco},
  year={2019},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/EscVM/Virtual_Security_Assistant}},
}
```
