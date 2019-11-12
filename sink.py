import cv2
import numpy as np
import math

def sinkMethod(img, showSteps = False):
    print("Sink method...")
    height, width = img.shape
    area = 255
    thresholdNo = 35
    lowerthreshold = 200
    lowestvalue = 0
    myset = set()
    threshImg = img.copy()
    threshImg[:, :] = lowestvalue
    threshImg[0, 0] = area
    myset.add((0, 0))
    while len(myset) != 0:
        currentY, currentX = myset.pop()
        if currentY - 1 >= 0 and threshImg[currentY - 1, currentX] == lowestvalue and img[currentY - 1, currentX] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY - 1, currentX])) <= thresholdNo:
            threshImg[currentY - 1, currentX] = area
            myset.add((currentY - 1, currentX))
        if currentY - 1 >= 0 and currentX - 1 >= 0 and threshImg[currentY - 1, currentX - 1] == lowestvalue and img[currentY - 1, currentX - 1] >= lowerthreshold and  abs(int(img[currentY, currentX]) - int(img[currentY - 1, currentX - 1])) <= thresholdNo:
            threshImg[currentY - 1, currentX - 1] = area
            myset.add((currentY - 1, currentX - 1))
        if currentX - 1 >= 0 and threshImg[currentY, currentX - 1] == lowestvalue and img[currentY, currentX - 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY, currentX - 1])) <= thresholdNo:
            threshImg[currentY, currentX - 1] = area
            myset.add((currentY, currentX - 1))
        if currentY + 1 < height and currentX - 1 >= 0 and threshImg[currentY + 1, currentX - 1] == lowestvalue and img[currentY + 1, currentX - 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY + 1, currentX - 1])) <= thresholdNo:
            threshImg[currentY + 1, currentX - 1] = area
            myset.add((currentY + 1, currentX - 1))
        if currentY + 1 < height and threshImg[currentY + 1, currentX] == lowestvalue and img[currentY + 1, currentX] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY + 1, currentX])) <= thresholdNo:
            threshImg[currentY + 1, currentX] = area
            myset.add((currentY + 1, currentX))
        if currentY + 1 < height and currentX + 1 < width and threshImg[currentY + 1, currentX + 1] == lowestvalue and img[currentY + 1, currentX + 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY + 1, currentX + 1])) <= thresholdNo:
            threshImg[currentY + 1, currentX + 1] = area
            myset.add((currentY + 1, currentX + 1))
        if currentX + 1 < width and threshImg[currentY, currentX + 1] == lowestvalue and img[currentY, currentX + 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY, currentX + 1])) <= thresholdNo:
            threshImg[currentY, currentX + 1] = area
            myset.add((currentY, currentX + 1))
        if currentY - 1 >= 0 and currentX + 1 < width and threshImg[currentY - 1, currentX + 1] == lowestvalue and img[currentY - 1, currentX + 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY - 1, currentX + 1])) <= thresholdNo:
            threshImg[currentY - 1, currentX + 1] = area
            myset.add((currentY - 1, currentX + 1))

    threshImg = cv2.threshold(threshImg, 128, 255, cv2.THRESH_BINARY)[1]

    if showSteps:
       cv2.imshow('Sink', threshImg)
       cv2.waitKey(0)

    return threshImg


def getLinesAngle(image, showUI = False):
	kern = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

	linedetectimg = image.copy()
	for i in range(0, 5):
		linedetectimg = cv2.erode(linedetectimg, kern)

	if showUI:
		cv2.imshow("erode", linedetectimg)

	linedetectimg = cv2.Canny(linedetectimg, 100, 200, None, 5, True)
	linesP = cv2.HoughLinesP(linedetectimg, 1, np.pi / 180, 50, None, 50, 10)
	linedetectimg = cv2.cvtColor(linedetectimg, cv2.COLOR_GRAY2BGR)
	l = linesP[0][0]
	cv2.line(linedetectimg, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)

	return (math.atan((l[1] - l[3]) / (l[0] - l[2])) * 180 / math.pi)