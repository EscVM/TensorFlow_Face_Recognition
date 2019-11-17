<h1 align="center"> ~ Virtual Security Assistant ~ </h1>

Have you ever desired a personal tireless [keeper](https://media.urbanpost.it/wp-content/uploads/2019/03/ScreenShot2016-01-19at4.18.36PM.jpg) able to recognize people passing through your gate and inform you whenever you want? Do you want to remove badges from your office and automatically and precise take note of all people coming in and out? Or, do you want to create your personal "[big brother](https://qph.fs.quoracdn.net/main-qimg-d30d05225145292c4e9186a4c8fac778)" world in your neighborhood?
If so, this repository is right for you. Powered by a cascade of four state of the art very stupid and completely unaware neural networks, this repository is giving you the possibility to recognize with a cheap and low range camera all persons transit in its field of view. Moreover, all data is saved and stored in a practical ".json" file that can be used for whatever you want. For example, we created a very simple server able to parse this file and interact with Google Assistant. So, when everything is set, you will be able to wake up your google assistant from whenever device you like and ask who have been spotted by your Virtual Security Assistant. 

All code have been tested on a NVDIA 2080 that can easilly give you more than 40 fps. Moreover, it perfectly works also on a [Jetson Xavier](https://www.nvidia.com/it-it/autonomous-machines/embedded-systems/jetson-agx-xavier/) that can be easily installed on the site where you want to deploy your sistem (let's move this AI on the edge).

**Side Notes**
This is repository is a stupid demo made in a hurry for our robotic center by me and ?. It has been a distraction from our research, but that with the community help can really become an interesting and helpful project. It is based on [dbib](http://dlib.net/) and heavily inspired by [face_recognition](https://github.com/ageitgey/face_recognition) repository. Whover has worked with "face_recognition" repository knows that it does not work well and is far from been reliable and accurate. We improved the general framework and put everything together in sweet and compact ready to work system. We intentionally left a easy customizable framework in order to let the community work and improve this Virtual Security Assistant.
