# Script based off of the depthAI versions for solely the mono wide camera (non depthAI)
# Team 935
#
# With Guidence
#
# If time
# TODO: Add DepthAI camera object
#
# Questions
# How to identify the different camera ids
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

# uses a video Capture and string 
def processApriltags(cap, nt_name):
#    print("processing " + nt_name)
    hasData, frame = cap.read()
    inputImageTime = time.monotonic()
    tag_state = []
    tag_state = ["LOST" for i in range(16)]
    if (hasData):
        print("processing has data" + nt_name)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Feed gray scale image into AprilTag library
        tag_info = detector.detect(gray)
        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]

        # OPTIONAL: Ignore any tags not in the set used on the 2024 FRC field:
        #filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 9))]

        for tag in filter_tags:

            est = estimator.estimateOrthogonalIteration(tag, DETECTION_ITERATIONS)
            pose = est.pose1
            tag_id = tag.getId()

            # Update Tag data in networktables
            if ((tag_id >= 1) & (tag_id <= 16)):
                tag_state[tag_id-1] = "TRACKED"
                ssd=sd.getSubTable(nt_name + f"/Tag[{tag_id}]")
                ssd.putString("Status", "TRACKED")
                ssd.putNumberArray("Pose", [pose.translation().x, pose.translation().y, pose.translation().z, pose.rotation().x, pose.rotation().y, pose.rotation().z])
    else:
        print("processing has NO data" + nt_name)

    # Update Tag data in networktables
    for i in range(16):
        if tag_state[i] == "LOST" :
            ssd=sd.getSubTable(nt_name + f"/Tag[{i+1}]")
            ssd.putString("Status", "LOST")
            ssd.putNumberArray("Pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Update Latency data in networktables
    latency2 = time.monotonic() - inputImageTime
    ssd=sd.getSubTable(nt_name + "/Latency")
    ssd.putNumber("Apriltag", float(latency2))
    ntinst.flush()

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

    cam1Name = "FrontCam"
    cam2Name = "BackCam"

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    print(f"{cap1}")
    print(f"{cap2}")

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

    while True:

        processApriltags(cap1, cam1Name)
        processApriltags(cap2, cam2Name)

        if cv2.waitKey(1) == ord('q'):
            break
    
    # After the loop release the cap object
    input.release()