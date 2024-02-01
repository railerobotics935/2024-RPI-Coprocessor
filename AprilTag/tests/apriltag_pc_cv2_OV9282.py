#!/usr/bin/env python3

import cv2

import robotpy_apriltag
from wpimath.geometry import Transform3d
import math


width = 1280
height = 800

# open video0, adding CAP_DSHOW is required on the PC to get higher resolution output
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("******************************")

print("Width = ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height = ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Auto Focus = ",cap.get(cv2.CAP_PROP_AUTOFOCUS))
print("Auto Exposure = ",cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
print("Auto WB = ",cap.get(cv2.CAP_PROP_AUTO_WB))

print("Brightness = ",cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("Contrast = ",cap.get(cv2.CAP_PROP_CONTRAST))
print("Saturation = ",cap.get(cv2.CAP_PROP_SATURATION))
print("Gain = ",cap.get(cv2.CAP_PROP_GAIN))
print("Hue = ",cap.get(cv2.CAP_PROP_HUE))
print("Exposure = ",cap.get(cv2.CAP_PROP_EXPOSURE))
print("WB Temp = ",cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
print("******************************")

# The control range can be viewed through v4l2-ctl -L
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
#cap.set(cv2.CAP_PROP_AUTO_WB, -1.0)
#cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 10)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 64)
#cap.set(cv2.CAP_PROP_CONTRAST, 100)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("Auto Focus = ",cap.get(cv2.CAP_PROP_AUTOFOCUS))
print("Auto Exposure = ",cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
print("Auto WB = ",cap.get(cv2.CAP_PROP_AUTO_WB))

print("Brightness = ",cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("Contrast = ",cap.get(cv2.CAP_PROP_CONTRAST))
print("Saturation = ",cap.get(cv2.CAP_PROP_SATURATION))
print("Gain = ",cap.get(cv2.CAP_PROP_GAIN))
print("Hue = ",cap.get(cv2.CAP_PROP_HUE))
print("Exposure = ",cap.get(cv2.CAP_PROP_EXPOSURE))
print("WB Temp = ",cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
print("******************************")

#print("POS msec = ",cap.get(cv2.CAP_PROP_POS_MSEC))
#print("******************************")

"""
Auto Focus =  -1.0
Auto Exposure =  0.0
Auto WB =  -1.0
Brightness =  0.0
Contrast =  50.0
Saturation =  50.0
Gain =  -1.0
Hue =  0.0
Exposure =  -6.0

cv2.CAP_PROP_APERTURE
cv2.CAP_PROP_ARAVIS_AUTOTRIGGER
cv2.CAP_PROP_AUTOFOCUS
cv2.CAP_PROP_AUTO_EXPOSURE
cv2.CAP_PROP_AUTO_WB
cv2.CAP_PROP_BACKEND
cv2.CAP_PROP_BACKLIGHT
cv2.CAP_PROP_BITRATE
cv2.CAP_PROP_BRIGHTNESS
cv2.CAP_PROP_BUFFERSIZE
cv2.CAP_PROP_CHANNEL
cv2.CAP_PROP_CODEC_PIXEL_FORMAT
cv2.CAP_PROP_CONTRAST
cv2.CAP_PROP_CONVERT_RGB
cv2.CAP_PROP_EXPOSURE
cv2.CAP_PROP_EXPOSUREPROGRAM
cv2.CAP_PROP_FOCUS
cv2.CAP_PROP_FORMAT
cv2.CAP_PROP_FOURCC
cv2.CAP_PROP_FPS
cv2.CAP_PROP_FRAME_COUNT
cv2.CAP_PROP_FRAME_HEIGHT
cv2.CAP_PROP_FRAME_WIDTH
cv2.CAP_PROP_GAIN
cv2.CAP_PROP_GAMMA
cv2.CAP_PROP_HUE
cv2.CAP_PROP_IRIS
cv2.CAP_PROP_ISO_SPEED
cv2.CAP_PROP_MODE
cv2.CAP_PROP_MONOCHROME
cv2.CAP_PROP_POS_AVI_RATIO
cv2.CAP_PROP_POS_FRAMES
cv2.CAP_PROP_POS_MSEC
cv2.CAP_PROP_RECTIFICATION
cv2.CAP_PROP_ROLL
cv2.CAP_PROP_SAR_DEN
cv2.CAP_PROP_SAR_NUM
cv2.CAP_PROP_SATURATION
cv2.CAP_PROP_SETTINGS
cv2.CAP_PROP_SHARPNESS
cv2.CAP_PROP_SPEED
cv2.CAP_PROP_WB_TEMPERATURE
cv2.CAP_PROP_WHITE_BALANCE_BLUE_U
cv2.CAP_PROP_WHITE_BALANCE_RED_V
"""

#print("******************************")
#print("Width = ",cap.get(FRAME_WIDTH))
#print("Height = ",cap.get(FRAME_HEIGHT))
#print("Framerate = ",cap.get(FRAME_RATE))
#print("Brightness = ",cap.get(BRIGHTNESS))
#print("Contrast = ",cap.get(CONTRAST))
#print("Saturation = ",cap.get(SATURATION))
#print("Gain = ",cap.get(GAIN))
#print("Hue = ",cap.get(HUE))
#print("Exposure = ",cap.get(EXPOSURE))
#print("******************************")  

# April Tag detection:
detector = robotpy_apriltag.AprilTagDetector()
detector.addFamily("tag36h11")
# see https://robotpy.readthedocs.io/projects/robotpy/en/stable/robotpy_apriltag/AprilTagDetector.html
#    detector.Config.debug = True                                # writes images to current working directory? did not work
#    detector.Config.decodeSharpening = 0.5                      # default is 0.25
#    detector.Config.numThreads = 1                              # default is 1
detector.Config.quadDecimate = 1.0                          # default is 2.0
detector.Config.quadSigma = 0.5                             # default is 0.0
#    detector.QuadThresholdParameters.criticalAngle = 0.0        # default is 10 degrees, unit in radians
detector.QuadThresholdParameters.maxLineFitMSE = 30.0       # default is 10.0
detector.QuadThresholdParameters.maxNumMaxima = 5           # default is 10     This made the biggest difference
detector.QuadThresholdParameters.minClusterPixels = 3       # default is 5
detector.QuadThresholdParameters.minWhiteBlackDiff = 0      # default is 5
estimator = robotpy_apriltag.AprilTagPoseEstimator(
    robotpy_apriltag.AprilTagPoseEstimator.Config(
        0.165, 780, 780, width / 2.0, height / 2.0
    )
)

# Detect apriltag
DETECTION_MARGIN_THRESHOLD = 10
DETECTION_ITERATIONS = 50

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to gray frame for Apriltag library
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Feed gray frame into Apriltag detector
    tag_info = detector.detect(gray)
    filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]

    # OPTIONAL: Ignore any tags not in the set used on the 2023 FRC field:
    #filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 9))]

    for tag in filter_tags:

        est = estimator.estimateOrthogonalIteration(tag, DETECTION_ITERATIONS)
        pose = est.pose1

        tag_id = tag.getId()
        center = tag.getCenter()
        #hamming = tag.getHamming()
        decision_margin = tag.getDecisionMargin()
        print(f"{tag_id}: {decision_margin} {pose}")

        # Highlight the edges of all recognized tags and label them with their IDs:

        if ((tag_id > 0) & (tag_id < 9)):
            col_box = (0,255,0)
            col_txt = (255,255,255)
        else:
            col_box = (0,0,255)
            col_txt = (0,255,255)

        # Draw a frame around the tag:
        corner0 = (int(tag.getCorner(0).x), int(tag.getCorner(0).y))
        corner1 = (int(tag.getCorner(1).x), int(tag.getCorner(1).y))
        corner2 = (int(tag.getCorner(2).x), int(tag.getCorner(2).y))
        corner3 = (int(tag.getCorner(3).x), int(tag.getCorner(3).y))
        cv2.line(frame, corner0, corner1, color = col_box, thickness = 2)
        cv2.line(frame, corner1, corner2, color = col_box, thickness = 2)
        cv2.line(frame, corner2, corner3, color = col_box, thickness = 2)
        cv2.line(frame, corner3, corner0, color = col_box, thickness = 2)

        # Label the tag with the ID:
        cv2.putText(frame, f"{tag_id}", (int(center.x), int(center.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, col_txt, 2)

        # print performance metrics
#        latency2 = dai.Clock.now() - inFrame.getTimestamp()
#        print(f"{latency1}; {latency2}")

#    cv2.putText(frame, "Fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

#    cv2.imshow("mono", frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)
#    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()