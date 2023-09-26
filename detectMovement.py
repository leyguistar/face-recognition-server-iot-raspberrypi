import socket
import pyttsx3
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
	data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
	res = data.decode('ascii')
	engine = pyttsx3.init()
	engine.say(res)
	engine.runAndWait()