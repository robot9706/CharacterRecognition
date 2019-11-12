import cv2
import argparse

import recognition
import sink

from tkinter import Tk
from tkinter.filedialog import askopenfilename

def thresh(im, sinkMethod=False, showSteps = False):
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	if sinkMethod:
		print("Threshold: Sink")
		thr = sink.sinkMethod(gray, showSteps)
	else:
		print("Threshold: Basic")
		thVal, thr = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

	return thr

def lineDetect(fileName, sinkMethod=False, maxDist=30, maxAngle=30, debug=False):
	print("File:", fileName)
	img = cv2.imread(fileName)
	cv2.imshow("Input", img)

	thr = thresh(img,sinkMethod=sinkMethod,showSteps=debug)

	print("Calculating base angle...")
	angle = sink.getLinesAngle(thr,showUI=debug)
	print("Base angle: ", angle)

	proc = recognition.doRecognition(thr,maxDist=maxDist, maxAngle=maxAngle, baseAngle=-angle)
	cv2.imshow("Output", proc)
	cv2.waitKey(0)


# Main
parser = argparse.ArgumentParser("main")
parser.add_argument("-s", "--sink",action="store_true")
parser.add_argument("-d", "--dist",nargs='?',const=30,type=int,default=30)
parser.add_argument("-a", "--angle",nargs='?',const=30,type=int,default=30)
parser.add_argument("-f", "--file",type=str,default="")
parser.add_argument("--debug",action="store_true")
args = parser.parse_args()

root = Tk()
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

fileName = args.file
if fileName == "" or fileName == None or len(fileName) == 0:
	fileName = askopenfilename()
lineDetect(fileName, args.sink, args.dist, args.angle, args.debug)