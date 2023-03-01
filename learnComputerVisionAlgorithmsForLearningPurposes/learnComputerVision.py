import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import math
# ###Lesson 1 Reading and writing an image

# img_grayscale = cv2.imread("test.jpg",0) #0 flag means read image in grayscale mode

# cv2.imshow('grayscale', img_grayscale) #name, show what the image to screen

# cv2.waitKey(0) #dont destroy the image on screen until a key is pressed

# cv2.destroyAllWindows()

# cv2.imwrite('grayscale.jpg', img_grayscale)


#opencv reads color images in BGR format, not RGB format which is helpful for other libraries




###Lesson 2: Annotating an image
# img = cv2.imread("test.jpg")
# if img is None:
#     print("cant read img")

# imageLine = img.copy() #create a copy of the original image so we can change the copy without changing the og
# #draw the line from point A to point B
# pointA = (200,80)
# pointB = (450,180)

# #line(whatImageVar,startpoint,end_point,color(bgr),thickness)
# cv2.line(imageLine,pointA, pointB,(255,0,0), thickness=3, lineType=cv2.LINE_AA)

# #circle(image,center_coord,radius,color,thickness=-1 for filled in circle)
# circle_center = (150,150)
# radius = 60
# cv2.circle(imageLine,circle_center,radius,(0,255,0),thickness=3, lineType=cv2.LINE_AA)

# #rectangle
# start_point = (200,300)
# end_point = (400,500)
# cv2.rectangle(imageLine,start_point,end_point,(0,0,255),thickness = 3, lineType = cv2.LINE_AA)

# #ellipse,half ellipse, other shapes

# #text
# text = 'This is a plant'
# location = (450,400)
# cv2.putText(imageLine,text,location,fontFace= cv2.FONT_HERSHEY_COMPLEX,fontScale = 1.5, color = (250,225,100))

# cv2.imshow('imageLine', imageLine)
# cv2.waitKey(0)

# print("okay")



#Lesson 3 #Edge detection

# img = cv2.imread("test.jpg", flags=0) #color not needed for edge detection

# img_blur = cv2.GaussianBlur(img,(3,3), 0) 

# cv2.imshow("image",img)
# cv2.imshow("image2",img_blur)



# #sobel edge detection

# sobelx = cv2.Sobel(src=img_blur, ddepth = cv2.CV_64F,dx=1,dy=0,ksize=5)
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
 

# cv2.imshow('sobelx', sobelx)
# cv2.imshow('sobely', sobely)
# cv2.imshow('sobelz', sobelxy)


# edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
# cv2.imshow("canny", edges)
# cv2.waitKey(0)

# #lesson 4 blob detection
# # Read image
# im = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)
 
# # Set up the detector with default parameters.
# detector = cv2.SimpleBlobDetector()
 
# # Detect blobs.
# keypoints = detector.detect(im)
 
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)

#annotating an image with the mouse in real time
# # Lists to store the bounding box coordinates
# top_left_corner=[]
# bottom_right_corner=[]
 
# # function which will be called on mouse input
# def drawRectangle(action, x, y, flags, *userdata):
#   # Referencing global variables 
#   global top_left_corner, bottom_right_corner
#   # Mark the top left corner when left mouse button is pressed
#   if action == cv2.EVENT_LBUTTONDOWN:
#     top_left_corner = [(x,y)]
#     # When left mouse button is released, mark bottom right corner
#   elif action == cv2.EVENT_LBUTTONUP:
#     bottom_right_corner = [(x,y)]    
#     # Draw the rectangle
#     cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,255,0),2, 8)
#     cv2.imshow("Window",image)
 
# # Read Images
# image = cv2.imread("test.jpg")
# # Make a temporary image, will be useful to clear the drawing
# temp = image.copy()
# # Create a named window
# cv2.namedWindow("Window")
# # highgui function called when mouse events occur
# cv2.setMouseCallback("Window", drawRectangle)
 
# k=0
# # Close the window when key q is pressed
# while k!=113:
#   # Display the image
#   cv2.imshow("Window", image)
#   k = cv2.waitKey(0)
#   # If c is pressed, clear the window, using the dummy image
#   if (k == 99):
#     image= temp.copy()
#     cv2.imshow("Window", image)
 
# cv2.destroyAllWindows()

def resize(img):
        return cv2.resize(img,(1024,512)) # arg1- input image, arg- output_width, output_height



def test1():
        cap = cv2.VideoCapture("lift1.mp4")

        #object detection from a stable camera
        object_detector = cv2.createBackgroundSubtractorMOG2()

        while True:
                ret,frame = cap.read()
                height,width, _ = frame.shape
                
                #extract region of interest
                roi = frame[240:1050,800:1100]

                mask = object_detector.apply(roi)
                _, mask = cv2.threshold(mask,250,255,cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                        #calc area and remove small elements
                        area = cv2.contourArea(cnt)
                        if area > 110:
                                #cv2.drawContours(roi, [cnt],-1,(0,255,0), 2)
                                x,y,w,h = cv2.boundingRect(cnt)
                                cv2.rectangle(roi,(x,y), (x+w,y+h), (0,255,0),2)
                cv2.imshow("frame",resize(frame))
                cv2.imshow("mask",resize(mask))
                #cv2.imshow("roi",roi)
                key = cv2.waitKey(30)

                if key == 113:
                        break
        cap.release()
        cv2.destroyAllWindows()



def findNextBlackPoint(basePoint, searchArea):
        square = 50
        #print(basePoint)
        radius = 30
        endPoint = [basePoint[0]+square,basePoint[1] + square]
        #cv2.circle(searchArea,basePoint,radius,(0,255,0),thickness=3, lineType=cv2.LINE_AA)
        cv2.rectangle(searchArea,basePoint,endPoint,(0,0,255),thickness = 3, lineType = cv2.LINE_AA)
        lowestPixelValue = 90000000
        for xx in range(-20,20):
                for yy in range(-20,20):
                        
                        #this is horizontal coloring
                        totalPixelValue = 0
                        totalPixelValue = searchArea[basePoint[1]+xx:basePoint[1] + square+xx,basePoint[0]+yy:basePoint[0] + square+yy].sum()
                        # for j in range(basePoint[1]+xx,basePoint[1] + 20+xx):
                        #         pass
                        #         block = searchArea[j]
                        #         totalPixelValue += block[basePoint[0]+yy:basePoint[0] + 20+yy].sum()
                                #for i in range(basePoint[0]+yy,basePoint[0] + 20+yy):
                                        #block[i][2] = 0
                                        #block[i][1] = 0
                                        #block[i][0] = 0
                                        #totalPixelValue += block[i].sum()
                        if totalPixelValue < lowestPixelValue:
                                lowestPixelValue = totalPixelValue
                                x = basePoint[0] + yy
                                y = basePoint[1] + xx
        #print(totalPixelValue)
        

        return x,y






def test3():
        print("hi")
        solutions = mp.solutions.pose

        #cv2.line(imageLine,pointA, pointB,(255,0,0), thickness=3, lineType=cv2.LINE_AA)
        # img = cv2.imread("test.jpg")
        # if img is None:
        #         print("cant read img")

        # imageLine = img.copy() #create a copy of the original image so we can change the copy without changing the og
        # pointA = (200,80)
        # pointB = (450,480)
        # cv2.line(imageLine,pointA, pointB,(255,0,0), thickness=3, lineType=cv2.LINE_AA)

        # cv2.imshow('line', imageLine)
        # cv2.waitKey(0)

        cap = cv2.VideoCapture('lift1.mp4')
        pointA = [775,875]
        pointB = [770,880]
        pointC = [770,870]
        lines = [pointA,pointB,pointC]
        while True:
                ret,frame = cap.read()
                if type(frame) == type(None):
                        print("the end")
                        break
                height,width,_ = frame.shape
                
                roi = frame[140:1080,200:1100]
                
                
                for i in range(len(lines)-1):
                        x1 = lines[i][0] + 25
                        y1 = lines[i][1] + 25
                        x2 = lines[i+1][0] + 25
                        y2 = lines[i+1][1] + 25
                        pt1 = [x1,y1]
                        pt2 = [x2,y2]
                        #cv2.line(roi,lines[i],lines[i+1],(255,0,0), thickness=3,lineType=cv2.LINE_AA)
                        cv2.line(roi,pt1,pt2,(255,0,0), thickness=3,lineType=cv2.LINE_AA)
                #cv2.line(roi,pointB,pointC,(255,0,0), thickness=3,lineType=cv2.LINE_AA)
                x = np.random.randint(-3,3,size=(1,5))
                y = np.random.randint(-10,-5,size = (1,5))
                #nextPoint = [lines[-1][0] + x[0][0], lines[-1][1] + y[0][1]]
                nextPoint = findNextBlackPoint(lines[-1], roi)
                #print(nextPoint)
                #print(roi.shape)
                #this is horizontal coloring
                # block = roi[870]
                # for i in range(len(block)):
                #         block[i][2] = 0
                #         block[i][1] = 0
                #         block[i][0] = 0
                # block = roi[871]
                # for i in range(len(block)):
                #         block[i][2] = 0
                #         block[i][1] = 0
                #         block[i][0] = 0
                # block = roi[872]
                # for i in range(len(block)):
                #         block[i][2] = 0
                #         block[i][1] = 0
                #         block[i][0] = 0


                        #print("-------")
                        #print(block[770])
                        #block[770][1] = 0
                #roi[550][850][1] = 255
                #print("->",roi[550][850][1])
                #this is for vertical coloring
                # for block in roi:
                #         block[770][1] = 0
                #         block[770][0] = 0
                #         block[770][2] = 0
                #first block is left,right. Second value is up down
                circle_center = (775,875)
                radius = 40
                #cv2.circle(roi,circle_center,radius,(0,255,0),thickness=3, lineType=cv2.LINE_AA)

                lines.append(nextPoint)
                cv2.imshow('frame1',resize(frame))
                key = cv2.waitKey(3)
                
                
                if key == 113:
                        break

        cap.release()
        cv2.destroyAllWindows()



class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        #self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS

    def getPosition(self, img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList




# construct the argument parser and parse the arguments
def jointTracker():
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", required=True,
                help="path to our input video")
        ap.add_argument("-o", "--output", required=True,
                help="path to our output video")
        ap.add_argument("-s", "--fps", type=int, default=30,
                help="set fps of output video")
        ap.add_argument("-b", "--black", type=str, default=False,
                help="set black background")
        args = vars(ap.parse_args())


        pTime = 0
        black_flag = eval(args["black"])
        cap = cv2.VideoCapture(args["input"])
        out = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*"MJPG"), 
                        args["fps"], (int(cap.get(3)), int(cap.get(4))))

        detector = PoseDetector()

        while(cap.isOpened()):
                success, img = cap.read()
                
                if success == False:
                        break
                
                img, p_landmarks, p_connections = detector.findPose(img, False)
                
                # use black background
                if black_flag:
                        img = img * 0
                
                # draw points
                mp.solutions.drawing_utils.draw_landmarks(img, p_landmarks, p_connections)
                keypoints = []
                keypointNames = ["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"]
                keyPointNameIterator = 0
                dictionaryJointsByName = dict()
                for data_point in p_landmarks.landmark:
                        jointInfo = {
                                                'X': data_point.x,
                                                'Y': data_point.y,
                                                'Z': data_point.z,
                                                'Visibility': data_point.visibility,
                                                'JointName': keypointNames[keyPointNameIterator]
                                                }
                        keypoints.append(jointInfo)
                        
                        dictionaryJointsByName[keypointNames[keyPointNameIterator]] = jointInfo
                        keyPointNameIterator += 1
                
                #time.sleep(1)
                lmList = detector.getPosition(img)

                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                # This would print the angle between the given body joints
                # print(getAngle("left_elbow", "left_shoulder", "nose", keypoints))
                out.write(img)
                cv2.imshow("Image", img)
                cv2.waitKey(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()



def calculateAngle(joint1, joint2, joint3):
    # Get x, y coordinates of joints
    x1, y1 = joint1['X'], joint1['Y']
    x2, y2 = joint2['X'], joint2['Y']
    x3, y3 = joint3['X'], joint3['Y']

    # Calculate angle between three points using dot product
    # and arccosine formula
    a = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    b = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    c = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    angle = math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))

    return angle


def getAngle(jointName1, jointName2, jointName3, keypoints):
    keypointNames = ["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"]
    jointIndex1 = keypointNames.index(jointName1)
    jointIndex2 = keypointNames.index(jointName2)
    jointIndex3 = keypointNames.index(jointName3)

    joint1 = keypoints[jointIndex1]
    joint2 = keypoints[jointIndex2]
    joint3 = keypoints[jointIndex3]


    angle = calculateAngle(joint1, joint2, joint3)

    return angle

        
def test4():
        print("test 4")









#test3()
#test4()
jointTracker()#in terminal run->>> python learnComputerVision.py -i deadliftFront.mp4 -o output_videos.mp4 -b False











