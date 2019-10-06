import cv2
import numpy as np
from random import Random
import math

random = Random()

def thresh(im):
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	#thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
	thVal, thr = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

	return thr


def random_color():
	return (int(random.random() * 225) + 30, int(random.random() * 225) + 30, int(random.random() * 225) + 30)


def rect_inside(a, b):
	aleft, atop, aright, abottom = a
	bleft, btop, bright, bbottom = b

	return ((bleft >= aleft) and (btop >= atop) and (bright <= aright) and (bbottom <= abottom))


def rect_center(r):
	left, top, right, bottom = r
	return (left+((right - left) / 2), top + ((bottom - top) / 2))


def dist(p1, p2):
	p1x, p1y = p1
	p2x, p2y = p2
	return math.sqrt( ((p1x-p2x)**2)+((p1y-p2y)**2) )

def angle(p1, p2):
	p1x, p1y = p1
	p2x, p2y = p2
	return math.degrees(math.atan2(p2y - p1y, p2x - p1x)) % 360


# Returns [ (contour, rect)* ]
def mergecontours(list, xMax, yMax):
	proc = []

	# Create a (countour, rectangle, processed) tuple of each contour, where
	# rectangle is a (left, top, right, bottom) BB of the contour
	for cont in list:
		rl = xMax
		rt = yMax
		rr = 0
		rb = 0
		for p in cont:
			px = p[0][0]
			py = p[0][1]

			if px < rl: rl = px
			if px > rr: rr = px

			if py < rt: rt = py
			if py > rb: rb = py

		proc.append((cont, (rl, rt, rr, rb), False))

	# Go through each item and see if it contains any other item
	# If it does, the smaller BB is possibly a hole (ex. the inside of O), mark it as processed
	for i in range(0, len(proc)):
		contour, rect, processed = proc[i]
		if processed: continue

		for j in range(0, len(proc)):
			if i == j: continue

			contour2, rect2, processed2 = proc[j]

			if rect_inside(rect, rect2):
				proc[j] = (contour2, rect2, True)

	# Convert the item list into (contour, rectangle) tuples
	# Only return items which are not processed, these are very the BBs of letters
	result = []
	for contour, rect, processed in proc:
		if processed: continue

		result.append((contour, rect))

	return result


def findnear(list, fromIndex):
	near = []

	processList = []
	processList.append(fromIndex)

	while len(processList) > 0:
		procIndex = processList.pop()

		contour, rect, proc = list[procIndex]
		c1 = rect_center(rect)
		near.append((contour, rect))
		list[procIndex] = (contour, rect, True)

		minDist = -1
		minIndex = -1

		for i in range(0, len(list)):
			contour, rect, processed = list[i]
			if processed: continue

			# Find the distance
			c2 = rect_center(rect)

			d = dist(c1, c2)
			a = angle(c1, c2)
			if math.fabs(a) > 20: continue
			if d > 200: continue

			if minDist == -1 or (d < minDist):
				minDist = d
				minIndex = i

		if minIndex != -1:
			nextc, nextr, nexto = list[minIndex]
			near.append((nextc, nextr))
			list[minIndex] = (nextc, nextr, True)

			processList.append(minIndex)

	return near


def group_distangle(group1, group2):
	minDist = -1
	minAngle = -1

	for i in range(0, len(group1)):
		c1, r1 = group1[i]
		center1 = rect_center(r1)

		for j in range(0, len(group2)):
			c2, r2 = group2[j]

			center2 = rect_center(r2)

			d = dist(center1, center2)
			a = angle(center1, center2)

			if math.fabs(d) > 100: continue

			if (a < 20 and a > 340) or (a > 160 and a < 190):
				if minDist == -1 or (d < minDist):
					minDist = d
					minAngle = a

	return (minDist, minAngle)

def merge_groups(groups):
	for i in range(0, len(groups)):
		for j in range(0, len(groups)):
			if i == j: continue

			dist, angle = group_distangle(groups[i], groups[j])
			if dist != -1:
				newList = []
				for a in range(0, len(groups)):
					if a == i or a == j: continue
					newList.append(groups[a])
				newList.append(groups[i] + groups[j])

				return (newList, True)

	return (groups, False)


def findlines(list):
	# Convert the list
	groups = []
	for contour, rect in list:
		groups.append([(contour, rect)]) # False - not used

	# Process
	process = True
	while process:
		newGroups, modified = merge_groups(groups)
		if modified:
			groups = newGroups
		else:
			process = False

	return groups


def getareas(img):
	height, width = img.shape
	resultImage = np.zeros((height, width, 3), np.uint8)
	resultImage[:,:] = (255,255,255)

	# Invert the image and find contours
	invImage = (255 - img)
	contours, hierarchy = cv2.findContours(invImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	print("All contours: {}".format(len(contours)))

	# Merge contours (holes and letters ex. O)
	filtered = mergecontours(contours, width, height)
	print ("Filtered contours: {}".format(len(filtered)))

	# Debug drawing
	# for contour, rect in filtered:
	#	left, top, right, bottom = rect
	#	cv2.rectangle(resultImage, (left, top), (right, bottom), random_color(), cv2.FILLED)
	# for contour, rect in filtered:
	#	cv2.fillPoly(resultImage, pts=[contour], color=(0, 0, 0))

	# Find areas of letters
	areas = findlines(filtered)
	print("Groups: {}".format(len(areas)))

	for l in areas:
		clr = random_color()
		for contour, rect in l:
			left, top, right, bottom = rect
			cv2.rectangle(resultImage, (left, top), (right, bottom), clr, cv2.FILLED)
			cv2.fillPoly(resultImage, pts=[contour], color=(0, 0, 0))

	return resultImage


inputImage = cv2.imread("imgs/test_img.png")
cv2.imshow("Input", inputImage)

thrImage = thresh(inputImage)
cv2.imwrite("thr.png", thrImage)

res = getareas(thrImage)

cv2.imshow("Output", res)

cv2.waitKey(0)