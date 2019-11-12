import cv2
import numpy as np
from random import Random
import math

# TOOLS
random = Random()

def random_color():
	return (int(random.random() * 205) + 50, int(random.random() * 205) + 50, int(random.random() * 205) + 50)

def rect_inside(a, b):
	aleft, atop, aright, abottom = a
	bleft, btop, bright, bbottom = b

	return ((bleft >= aleft) and (btop >= atop) and (bright <= aright) and (bbottom <= abottom))


def rect_center(r): # Returns (X,Y)
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


# STEP 2 - CONTOUR FILTER
def mergecontours(list, xMax, yMax): # Returns [ (contour, rect)* ]
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


# STEP 3 - CREATE TILES
class TiledImage:
	tiles = [] # Packed 2D array, elements: [ [ (contour, rect, originalIndex)* ]* ]
	numX = 0
	numY = 0
	tileSize = 0

	def __init__(self, tilesX, tilesY, tileSize, contours): # contours = [ (contour, rect)* ]
		self.numX = tilesX
		self.numY = tilesY
		self.tileSize = tileSize

		# Init the array
		num = self.numX * self.numY
		for i in range(0, num):
			self.tiles.append([]) # Append an empty array @ index i.

		# Put contours into tiles
		for i in range(0, len(contours)):
			contour, rect = contours[i]
			x, y = rect_center(rect)

			tx, ty = self.tileAt(x, y)
			if tx < 0 or ty < 0 or tx >= self.numX or ty >= self.numY:
				print ("Contour skipped, out of bounds!")
				continue

			offset = self.tileOffset(tx, ty)
			self.tiles[offset].append((contour, rect, i))

	def tileOffset(self, tx, ty):
		return tx + ty * self.numX

	def tileAt(self, pixelX, pixelY):
		return (math.floor(pixelX / self.tileSize), math.floor(pixelY / self.tileSize))

	def getTile(self, tx, ty):
		if tx < 0 or ty < 0 or tx >= self.numX or ty >= self.numY:
			return []

		return self.tiles[self.tileOffset(tx, ty)]

	def getNearby(self, tx, ty):
		return self.getTile(tx, ty) + self.getTile(tx + 1, ty) + self.getTile(tx, ty - 1) + self.getTile(tx + 1, ty - 1) + self.getTile(tx, ty + 1) + self.getTile(tx + 1, ty + 1) + self.getTile(tx - 1, ty) + self.getTile(tx - 1, ty + 1) + self.getTile(tx - 1, ty - 1)


def findlines(list, thrImage, maxDistance, maxAngle, baseAngle):
	height, width = thrImage.shape
	tileSize = 200
	tiles = TiledImage(math.ceil(width / tileSize), math.ceil(height / tileSize), tileSize, list)

	# Calculate max degrees
	minDeg1 = (180 - baseAngle - maxAngle)
	maxDeg1 = (180 - baseAngle + maxAngle)

	minDeg2 = (360 - baseAngle - maxAngle)
	maxDeg2 = -baseAngle + maxAngle

	# Start processing tiles
	groups = [ -1 for i in range(len(list)) ] # Indexed by contour indexes, each element = the group index of that element
	nextGroup = 0

	for tileX in range(0, tiles.numX):
		for tileY in range(0, tiles.numY):
			tile = tiles.getTile(tileX, tileY)
			if len(tile) == 0:
				continue

			nearby = tiles.getNearby(tileX, tileY)

			for contour, rect, index in tile:
				#if groups[index] > -1:
				#	continue # Item is already processed

				c1 = rect_center(rect)

				# Put the element into a new group
				groups[index] = nextGroup
				nextGroup = nextGroup + 1

				# Find other elements
				for other in nearby:
					otherContour, otherRect, otherIndex = other
					if otherIndex == index:
						continue # Already processed or it's the same item

					c2 = rect_center(otherRect)

					d = dist(c1, c2)
					a = angle(c1, c2)

					if d < maxDistance: # Distance check
						if (a > minDeg1 and a < maxDeg1) or (a > minDeg2 or a < maxDeg2): # Angle check
							# Merge
							if groups[otherIndex] > -1: # The element is already in a group, merge them
								oldGroup = groups[otherIndex]
								newGroupIndex = groups[index]
								for gIndex in range(0, len(groups)):
									if groups[gIndex] == oldGroup:
										groups[gIndex] = newGroupIndex
							else: # The element is not in a group
								groups[otherIndex] = groups[index]

	return groups


def getareas(img, maxDist, maxAngle, baseAngle):
	print("Processing image...")

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

	# Find areas of letters
	areas = findlines(filtered, img, maxDist, maxAngle, baseAngle)
	distGroups = set(areas)
	print("Groups: {}".format(len(distGroups)))

	print("Debug rendering...")

	# Pre process and draw groups
	groupColors = [0 for i in range(0, len(areas))]
	for groupIndex in distGroups:
		groupColors[groupIndex] = random_color()

		left = width
		top = height
		right = 0
		bottom = 0

		for elemIndex in range(0, len(filtered)):
			if not areas[elemIndex] == groupIndex:
				continue

			contour, rect = filtered[elemIndex]
			eleft, etop, eright, ebottom = rect

			if eleft < left:
				left = eleft
			if etop < top:
				top = etop
			if eright > right:
				right = eright
			if ebottom > bottom:
				bottom = ebottom

		cv2.putText(resultImage, str(groupIndex), (left, top - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
		cv2.rectangle(resultImage, (left, top), (right, bottom), (100,100,100))

	# Draw group content

	for i in range(0, len(filtered)):
		contour, rect = filtered[i]
		group = areas[i]

		if group == -1:
			continue

		color = groupColors[group]

		left, top, right, bottom = rect
		cv2.rectangle(resultImage, (left, top), (right, bottom), color, cv2.FILLED)
		cv2.fillPoly(resultImage, pts=[contour], color=(0, 0, 0))

	print ("Done")

	return resultImage

# DO PROCESSING
def doRecognition(inputImage, maxDist = 30, maxAngle = 30, baseAngle = 0):
	res = getareas(inputImage, maxDist, maxAngle, baseAngle)
	cv2.imwrite("Output.png", res)

	return res


# INPUT
#process("imgs/test_14.png", 30, 30, 0) # Matek jelek :)

#process("imgs/test_4.png", 100, 30, 0) # Szöveg és kép

#process("imgs/test_img.png", 100, 30, 0) # Több méret és betűtípus

#process("imgs/ttt_deg58.png", 30, 30, 58) # Nem vízszint

#process("imgs/test_14.png", 30, 30, 0) # Több szöveg blokk
#process("imgs/test_14_deg_n67.png", 30, 30, -67) # Több szöveg blokk nem vízszint