import paho.mqtt.client as MQTT

class MyMQTT(object):
	def __init__(self, clientID, broker="iot.eclipse.org", MQTT_user=None, MQTT_pwd=None, notifier=None, port=1883):
		
		self.clientID = clientID
		# create an instance of paho.mqtt.client
		self._mqtt = MQTT.Client(self.clientID, False)
		self._mqtt.username_pw_set(username=MQTT_user,password=MQTT_pwd)
		
		# register the callbacks
		self._mqtt.on_connect = self.onConnect
		self._mqtt.on_message = self.onMessage
		self._mqtt.on_subscribe = self.onSubscribe
		self._mqtt.on_disconnect = self.onDisconnect
		
		self.notifier = notifier

		self.messageBroker = broker
		self.port = port
		
		self._topic = []
		self.is_subscribed = []
		self.is_connected = False

	
	def start(self):
		#manage connection to broker
		self._mqtt.connect(self.messageBroker, self.port)
		self._mqtt.loop_start()

	def stop(self):
		while self._topic:
			self.unsubscribe(self._topic[0])			
		self._mqtt.loop_stop()
		self._mqtt.disconnect()

	def publish(self, topic, message, qos=2):
		self._mqtt.publish(topic, message, qos)
		print(self.clientID,": published on topic " + topic + ":" + message) 
	
	def subscribe(self, topic, qos=2):
		if topic in self._topic:
			print(self.clientID,": already subscribed to topic %s." %(topic))
		else:
			self._mqtt.subscribe(topic, qos)
			self._topic.append(topic)
			self.is_subscribed.append(False)
	
	def unsubscribe(self, topic):
		if topic in self._topic:
			i = self._topic.index(topic)
			self._topic.remove(topic)
			self.is_subscribed.pop(i)
			self._mqtt.unsubscribe(topic)
			print(self.clientID,": unsubcribed by %s." %(topic))
	
	def onConnect (self, paho_mqtt, userdata, flags, rc):
		print(self.clientID,": connected to %s." % (self.messageBroker))
		self.is_connected = True
	
	def onDisconnect (self, paho_mqtt, userdata, rc):
		print(self.clientID,": disconnected from %s." % (self.messageBroker))
		self.is_connected = False
	
	def onSubscribe(self, client, userdata, mid, granted_qos):
		print(self.clientID,": subscribed to %s." %(self._topic[mid-1]))
		self.is_subscribed[mid-1] = True
	
	def onMessage(self, paho_mqtt, userdata, message):
		try: 
			self.notifier(message)
		except:
			pass