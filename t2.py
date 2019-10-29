import cv2
import numpy as np
from random import Random
import math
import queue

from tkinter import Tk
from tkinter.filedialog import askopenfilename

random = Random()

# Settings
thMode = 2  # 0 - BASIC, 1 - ADAPTIVE, 2 - SINKING
thLevel = 235

maxMergeDist = 3
apostArea = 15
angleGlob = 0


# Creates a filtered binary image
def threshold(inImg):
    global angleGlob
    if thMode == 0:
        thVal, thImg = cv2.threshold(inImg, thLevel, 255, cv2.THRESH_BINARY)
        return thImg
    elif thMode == 1:
        return cv2.adaptiveThreshold(inImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    elif thMode == 2:
        ret, angleGlob = sinkMethod(inImg, True)
        ret = sinkMethod(ret, False)[0]
        return ret

    return None


def sinkMethod(img, rotate):
    height, width = img.shape
    area = 255
    thresholdNo = 50
    lowerthreshold = 220
    lowestvalue = 0
    myQueue = queue.Queue(width * height)
    treshImg = img.copy()
    treshImg[:, :] = lowestvalue
    #for y in range(0, height):
     #   for x in range(0, width):
    if treshImg[0, 0] == lowestvalue:
        treshImg[0, 0] = area
        myQueue.put((0, 0))
        while not myQueue.empty():
            currentY, currentX = myQueue.get()
            treshImg[currentY, currentX] = area
            if currentY - 1 >= 0 and treshImg[currentY - 1, currentX] == lowestvalue and img[currentY - 1, currentX] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY - 1, currentX])) <= thresholdNo:
                treshImg[currentY - 1, currentX] = area
                myQueue.put((currentY - 1, currentX))
            if currentY - 1 >= 0 and currentX - 1 >= 0 and treshImg[currentY - 1, currentX - 1] == lowestvalue and img[currentY - 1, currentX - 1] >= lowerthreshold and  abs(int(img[currentY, currentX]) - int(img[currentY - 1, currentX - 1])) <= thresholdNo:
                treshImg[currentY - 1, currentX - 1] = area
                myQueue.put((currentY - 1, currentX - 1))
            if currentX - 1 >= 0 and treshImg[currentY, currentX - 1] == lowestvalue and img[currentY, currentX - 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY, currentX - 1])) <= thresholdNo:
                treshImg[currentY, currentX - 1] = area
                myQueue.put((currentY, currentX - 1))
            if currentY + 1 < height and currentX - 1 >= 0 and treshImg[currentY + 1, currentX - 1] == lowestvalue and img[currentY + 1, currentX - 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY + 1, currentX - 1])) <= thresholdNo:
                treshImg[currentY + 1, currentX - 1] = area
                myQueue.put((currentY + 1, currentX - 1))
            if currentY + 1 < height and treshImg[currentY + 1, currentX] == lowestvalue and img[currentY + 1, currentX] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY + 1, currentX])) <= thresholdNo:
                treshImg[currentY + 1, currentX] = area
                myQueue.put((currentY + 1, currentX))
            if currentY + 1 < height and currentX + 1 < width and treshImg[currentY + 1, currentX + 1] == lowestvalue and img[currentY + 1, currentX + 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY + 1, currentX + 1])) <= thresholdNo:
                treshImg[currentY + 1, currentX + 1] = area
                myQueue.put((currentY + 1, currentX + 1))
            if currentX + 1 < width and treshImg[currentY, currentX + 1] == lowestvalue and img[currentY, currentX + 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY, currentX + 1])) <= thresholdNo:
                treshImg[currentY, currentX + 1] = area
                myQueue.put((currentY, currentX + 1))
            if currentY - 1 >= 0 and currentX + 1 < width and treshImg[currentY - 1, currentX + 1] == lowestvalue and img[currentY - 1, currentX + 1] >= lowerthreshold and abs(int(img[currentY, currentX]) - int(img[currentY - 1, currentX + 1])) <= thresholdNo:
                treshImg[currentY - 1, currentX + 1] = area
                myQueue.put((currentY - 1, currentX + 1))
        #area = 1

    treshImg = cv2.threshold(treshImg, 128, 255, cv2.THRESH_BINARY)[1]
    # kern = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # treshImg = cv2.dilate(treshImg, kern)
    # treshImg = cv2.erode(treshImg, kern)
    # treshImg = cv2.erode(treshImg, kern)
    # treshImg = cv2.dilate(treshImg, kern)
    cv2.imwrite("balazs.png", img)
    cv2.imshow('Orig', cv2.resize(img, (width * 1, height * 1)))
    cv2.imwrite("balazs1.png", treshImg)
    cv2.imshow('Sink', cv2.resize(treshImg, (width * 1, height * 1)))
    angle = 0
    if rotate:
        coords = np.column_stack(np.where(treshImg == 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        treshImg = cv2.warpAffine(treshImg, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite("balazs2.png", treshImg)
        cv2.imshow('Rotate', cv2.resize(treshImg, (width * 1, height * 1)))
        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img2 = img.copy()
        img2 = cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)
        cv2.imshow('Rot', cv2.resize(img2, (width * 1, height * 1)))

    cv2.waitKey(0)
    cv2.destroyWindow('Orig')
    cv2.destroyWindow('Sink')
    if rotate:
        cv2.destroyWindow('Rotate')
        cv2.destroyWindow('Rot')
    return treshImg, angle


def safe_sample(img, x, y, w, h):
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0

    return int(img[y, x])


# Finds horizontal lines in an image, returns an array of line images
def getLines(inImg, showLines=False):
    lineStartY = -1

    lines = []

    height, width = inImg.shape
    for y in range(0, height):
        numBlackPixels = width - cv2.countNonZero(inImg[y])

        if numBlackPixels > 0:
            if lineStartY == -1:
                lineStartY = y

        if numBlackPixels <= 0:
            if lineStartY > -1:
                lines.append((lineStartY, y))
                lineStartY = -1

    if lineStartY > -1:
        lines.append((lineStartY, y))

    imgLines = []
    for i in range(0, len(lines)):
        startY, endY = lines[i]

        imgLines.append((inImg[startY:endY, 0:width], startY))

    if showLines:
        sy = 0
        for i in range(0, len(lines)):
            startY, endY = lines[i]
            cv2.rectangle(inImg, (0, sy + 1), (width, startY - 1), 200, cv2.FILLED)
            sy = endY + 1

        cv2.rectangle(inImg, (0, sy - 1), (width, height), 200, cv2.FILLED)

    return imgLines


# Checks if rectangle 'a' is inside rectangle 'b'
def rect_inside(a, b):
    aleft, atop, aright, abottom = a
    bleft, btop, bright, bbottom = b

    return ((bleft >= aleft) and (btop >= atop) and (bright <= aright) and (bbottom <= abottom))


# Finds the distance between two contours
def contour_dist(a, b):
    mindist = -1
    minp1 = [[-1, -1]]
    minp2 = [[-1, -1]]

    for c1 in a:
        for c2 in b:
            for p1 in c1:
                for p2 in c2:
                    dist = math.sqrt(((p1[0][0] - p2[0][0]) ** 2) + ((p1[0][1] - p2[0][1]) ** 2))
                    if mindist == -1 or dist < mindist:
                        mindist = dist
                        minp1 = p1
                        minp2 = p2

    dx = minp1[0][0] - minp2[0][0]
    dy = minp1[0][1] - minp2[0][1]

    if (abs(dx) > abs(dy)):
        return -1

    print((dx, dy))

    return mindist


# Merges two rectangles
def merge_rects(a, b):
    aleft, atop, aright, abottom = a
    bleft, btop, bright, bbottom = b

    return (min(aleft, bleft), min(atop, btop), max(aright, bright), max(abottom, bbottom))


# Merges contours which are really close to eachother
def merge_pass(data):
    preFilter = []
    for contour, rect in data:
        preFilter.append((contour, rect, False))

    # Check if rectangles are close to eachother
    # If might be possible that characters get split into smaller parts (ex. m)
    modified = False
    for i in range(0, len(preFilter)):
        contour, rect, processed = preFilter[i]
        if processed: continue

        for j in range(0, len(preFilter)):
            if i == j: continue

            contour2, rect2, processed2 = preFilter[j]
            if processed2: continue

        # area = 0
        # for c in contour2:
        #	area += cv2.contourArea(c)

        # ????????????????

        # dist = contour_dist(contour, contour2)
        # if dist == -1: continue

        # Check if a merge is possible
        # if dist < maxMergeDist:
        #	modified = True
        #	contour.extend(contour2) # Add the merged contour to the list
        #	preFilter[i] = (contour, merge_rects(rect, rect2), False) # False - Not merged into another rectangle
        #	preFilter[j] = (contour2, rect2, True) # True - merged into another rectangle

    if not modified:
        return (None, False)

    # Create the new return array
    result = []
    for contour, rect, processed in preFilter:
        if processed: continue

        result.append((contour, rect))

    return (result, True)


# Filters contour lists to find separated pixel regions
def filter_contours(list, xMax, yMax):
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
    preFilter = []
    for contour, rect, processed in proc:
        if processed: continue

        preFilter.append(([contour], rect))

    # Merge rectangles until none can be merged
    merged = preFilter
    merging = False  # TODO: Fix apostrophe merging

    while merging:
        mergeResult, modified = merge_pass(merged)
        if modified:
            merged = mergeResult
        else:
            merging = False

    return merged


# Returns a random color
def random_color():
    return (int(random.random() * 100) + 75, int(random.random() * 100) + 75, int(random.random() * 100) + 75)


# Processes a line of characters
def process_line(image):
    # Create an inverted version of the image
    invImage = image.copy()
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            invImage[x, y] = 255 - image[x, y]

    # Find contours in the inverted image
    contours, hierarchy = cv2.findContours(invImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours to find BBs of characters
    height, width = image.shape
    items = filter_contours(contours, width, height)

    # Draw the data
    colorImage = np.zeros((height, width, 3), np.uint8)
    colorImage[:, :] = (255, 255, 255)
    for contour, rect in items:
        left, top, right, bottom = rect
        cv2.rectangle(colorImage, (left, top), (right, bottom), (255, 178, 178), cv2.FILLED)
    # cv2.drawContours(colorImage, contour, -1, random_color(), -1)

    # Copy the real characters to BBs
    colorImage[image == 0] = (0, 0, 0)

    return colorImage


# Main
def processImage(path):
    global processFileName
    processFileName = path

    print(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read the input image
    cv2.imshow("Input", img)
    cv2.imshow("Output", img)

    filter = threshold(img)  # Filter the image
    lineImgs = getLines(filter, True)  # Separate the image into lines

    height, width = img.shape
    output = np.zeros((height, width, 3), np.uint8)
    output[:, :] = (200, 200, 200)

    for lineIndex in range(0, len(lineImgs)):
        image, y = lineImgs[lineIndex]

        processedLine = process_line(image)
        procHeight, procWidth, layers = processedLine.shape

        output[y:(y + procHeight), 0:procWidth] = processedLine

    global angleGlob
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, -angleGlob, 1.0)
    output = cv2.warpAffine(output, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow("Output", output)

    return


def ui_update(unused):
    global thMode
    thMode = cv2.getTrackbarPos("Filter mode", "Output")
    global thLevel
    thLevel = cv2.getTrackbarPos("Threshold", "Output")

    global processFileName
    processImage(processFileName)


# Main program
root = Tk()
root.withdraw()

# Create windows
cv2.namedWindow("Input", cv2.WINDOW_NORMAL)

cv2.namedWindow("Output")
cv2.createTrackbar("Filter mode", "Output", thMode, 2, ui_update)
cv2.createTrackbar("Threshold", "Output", thLevel, 255, ui_update)

# Do processing
processImage(askopenfilename())

cv2.waitKey()
cv2.destroyAllWindows()
