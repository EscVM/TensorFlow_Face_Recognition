#--------------------------
#Date: 19/12/2019
#Place: Turin, PIC4SeR
#Author: Fra, Vitto
#Project: faceAssistant
#---------------------------

from threading import Thread
import time

class Timer(Thread):  
    def __init__(self,name,t,function=None,repeat=True):
        Thread.__init__(self)
        self.daemon = True
        self.name = name
        self.repeat = repeat
        try:
            t = t.split('.')
            assert len(t) == 2
        except:
            try:
                t = t[0].split(':') # previous split turned it in a list
                assert len(t) == 2
            except:
                raise ValueError("Wrong time list: "+str(t))

        self.time = {"h": int(t[0]), "m":int(t[1])}
        self.function = function
        if self.time['h'] > 23 or self.time['h'] < 0:
            raise ValueError("Wrong hour: " + str(self.time['h']))
        elif self.time['m'] > 59 or self.time['m'] < 0:
            raise ValueError("Wrong minute: " + str(self.time['m']))
    
    
    
    def compute_delay(self):
        tm = time.time()
        sec = 60 - time.localtime(tm).tm_sec
        minut = self.time["m"] - 1 - time.localtime(tm).tm_min
        hour = self.time["h"] - (minut < 0) - time.localtime(tm).tm_hour
        if hour < 0:
            hour += 24
        if minut <0:
            minut += 60
        delay = sec + 60*(minut + hour*60)
        return tm + delay
    
    
    
    def run(self):
        while True:
            target = self.compute_delay()
            print("[INFO] Timer '{}' set to {}.".format(self.name,time.ctime(target)))
            time.sleep(max(0,target-time.time()))
            if self.function:
                self.function()
            print("[INFO] Timer '{}' has expired on {}.".format(self.name,time.ctime(target)))
            if not self.repeat:
                break