# Script based off of the depthAI versions for solely the mono wide camera (non depthAI)
# Team 935
# 
# DONE: Setup Pi configuration stuff
# DONE: Setup NT Communication
#
# With Guidence
# DONE: Setup Apriltag processing
# TODO: Debug process
# TODO: Process Apriltag
# TODO: Send Apriltag with latency to NT
#
# From Scratch
# TODO: Put it into a class
# TODO: Implement for multiple cameras
#
# If time
# TODO: Add DepthAI camera object
#

# test to use cv2 instead of samera server to 

# This will be used to set up the pi (Read the config file)
# =============================================
#from operator import truediv
from pathlib import Path
from cscore import CameraServer
from ntcore import NetworkTableInstance
from ntcore import NetworkTable

#for DepthAI processing
import json
import time
import sys
import cv2
import depthai as dai
import numpy as np

#for AprilTag processing
import robotpy_apriltag
from wpimath.geometry import Transform3d
import math
configFile = "/boot/frc.json"

team = 935
server = False

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readConfig():
    """Read configuration file."""
    global team
    global server

    # Parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # Top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # Team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    return True
# =============================================

# Define the class for a wide camera object
class WideCamera :
    # Constructor of class, set the name and id of the camera
    def __init__(camera, name, id):
        camera.name = name
        camera.idNum = id
    
    def processSomething(camera):
        pass
        

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTableInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClient4("wpilibpi")
        ntinst.setServerTeam(team)
        ntinst.startDSClient()
    sd=ntinst.getTable("SmartDashboard")

    # Constants
    # Width and Height of various image processing pipelines (optimized for speed and bandwidth)
    # OAK-D Lite mono camera -> 640x400 pixels
    width = 1280
    height = 800

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Create the apriltag detector
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
    #detector.QuadThresholdParameters.minWhiteBlackDiff = 0      # default is 5

    estimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            0.165, 780, 780, width / 2.0, height / 2.0
        )
    )
    # Detect apriltag
    DETECTION_MARGIN_THRESHOLD = 50
    DETECTION_ITERATIONS = 50

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    image_output_bandwidth_limit_counter = 0

    while True:
        test, frame = cap.read()
        inputImageTime = time.monotonic()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        # Feed gray scale image into AprilTag library
        tag_info = detector.detect(gray)
        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]

        # OPTIONAL: Ignore any tags not in the set used on the 2024 FRC field:
        #filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 9))]

        tag_state = []
        tag_state = ["LOST" for i in range(16)]
        for tag in filter_tags:

            est = estimator.estimateOrthogonalIteration(tag, DETECTION_ITERATIONS)
            pose = est.pose1

            tag_id = tag.getId()
            center = tag.getCenter()
            #hamming = tag.getHamming()
            #decision_margin = tag.getDecisionMargin()

            # Highlight the edges of all recognized tags and label them with their IDs:
            if ((tag_id >= 1) & (tag_id <= 16)):
                col_box = (0,255,0)
                col_txt = (0,255,0)
            else:
                col_box = (0,0,255)
                col_txt = (0,255,255)

            # Draw a frame around the tag
            corner0 = (int(tag.getCorner(0).x), int(tag.getCorner(0).y))
            corner1 = (int(tag.getCorner(1).x), int(tag.getCorner(1).y))
            corner2 = (int(tag.getCorner(2).x), int(tag.getCorner(2).y))
            corner3 = (int(tag.getCorner(3).x), int(tag.getCorner(3).y))
            cv2.line(frame, corner0, corner1, color = col_box, thickness = 1)
            cv2.line(frame, corner1, corner2, color = col_box, thickness = 1)
            cv2.line(frame, corner2, corner3, color = col_box, thickness = 1)
            cv2.line(frame, corner3, corner0, color = col_box, thickness = 1)

            # Label the tag with the ID:
            cv2.putText(frame, f"{tag_id}", (int(center.x), int(center.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, col_txt, 2)

            # Update Tag data in networktables
            if ((tag_id >= 1) & (tag_id <= 16)):
                tag_state[tag_id-1] = "TRACKED"
                ssd=sd.getSubTable(f"FrontCam/Tag[{tag_id}]")
                ssd.putString("Status", "TRACKED")
                ssd.putNumberArray("Pose", [pose.translation().x, pose.translation().y, pose.translation().z, pose.rotation().x, pose.rotation().y, pose.rotation().z])

        # Update Tag data in networktables
        for i in range(16):
            if tag_state[i] == "LOST" :
                ssd=sd.getSubTable(f"FrontCam/Tag[{i+1}]")
                ssd.putString("Status", "LOST")
                ssd.putNumberArray("Pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Update Latency data in networktables
        latency2 = time.monotonic() - inputImageTime
        ssd=sd.getSubTable("FrontCam/Latency")
        ssd.putNumber("Apriltag", float(latency2))
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)

        if cv2.waitKey(1) == ord('q'):
            break
    
    # After the loop release the cap object
    input.release()