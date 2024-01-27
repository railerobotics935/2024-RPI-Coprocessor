#!/usr/bin/env python3
#
# based on: https://docs.luxonis.com/projects/api/en/v2.1.0.0/samples/26_1_spatial_mobilenet/
# and: https://docs.luxonis.com/projects/api/en/latest/samples/ObjectTracker/spatial_object_tracker/#spatial-object-tracker-on-rgb
# updated to work on FRC 2024 WPILibPi image and to be uploaded as a vision application
# communicating with shuffleboard and RoboRIO through NetworkTables and CameraServer
# Jaap van Bergeijk, 2024

# Uses OAK-D Lite Color Camera with on camera ISP + Manipulator to size the image down 1/4 and convert to GRAY8
# - reduces USB data transfer time
# - takes camera ISP + Manipulator processing time
# - image resoluation is high enough for longer distance Apriltag detection (up to about 7 meters)

#from operator import truediv
from pathlib import Path
from cscore import CameraServer
from ntcore import NetworkTableInstance

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

    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
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
    # OAK-D Lite color camera ISP at 1/4, go from 4056x3040 -> 1014x760 pixels
    isp_num = 1
    isp_den = 4
    gray_width = 1014
    gray_height = 760

    # Create a CameraServer for ShuffleBoard visualization
    CameraServer.enableLogging()
#    camera = CameraServer.startAutomaticCapture()
#    cs.enableLogging()

    # Width and Height have to match Neural Network settings: 320x320 pixels
    output_stream_nn = CameraServer.putVideo("FrontNN", gray_width, gray_height)

    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = dai.Pipeline()

    # First, we want the Color camera as the output
    camRgb = pipeline.create(dai.node.ColorCamera)
    manipC2G = pipeline.create(dai.node.ImageManip)
    xoutGray = pipeline.create(dai.node.XLinkOut)

    xoutGray.setStreamName("gray")
       
    # Properties
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    camRgb.setInterleaved(False)
    camRgb.setIspScale(isp_num, isp_den)
    camRgb.setPreviewSize(gray_width, gray_height)
#    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(20)

    manipC2G.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)

    # Linking
    camRgb.preview.link(manipC2G.inputImage)
    manipC2G.out.link(xoutGray.input)

    # Create the apriltag detector
    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily("tag36h11")
#    detector.Config.quadDecimate = 1
    estimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            0.165, 780, 780, gray_width / 2.0, gray_height / 2.0
        )
    )
    # Detect apriltag
    DETECTION_MARGIN_THRESHOLD = 50
    DETECTION_ITERATIONS = 50

    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    #with dai.Device(pipeline, True) as device:
#    with dai.Device(pipeline, usb2Mode=True) as device:
    with dai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink

        # Output queues will be used to get the gray frames
        qGray = device.getOutputQueue(name="gray", maxSize=1, blocking=False)

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        image_output_bandwidth_limit_counter = 0

        while True:
            inGray = qGray.tryGet()

            if inGray is not None:
                latency1 = dai.Clock.now() - inGray.getTimestamp()

#                frame = inRgb.getCvFrame()
                grayFrame = inGray.getFrame()

#                print(f"{frame.shape[1]} : {frame.shape[0]}") # should be 416 x 416 (or nn_width x nn_height)
#                print(f"{depthFrame.shape[1]} : {depthFrame.shape[0]}") # should be 640 x 400

                if grayFrame is not None:
                    counter+=1
                    current_time = time.monotonic()
                    if (current_time - startTime) > 1 :
                        fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time

                    # Create color frame for Shuffleboard display
                    outFrame = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)

                    # Feed gray scale image into AprilTag library
                    tag_info = detector.detect(grayFrame)
                    apriltags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
                    # Ignore any tags not in the set used on the 2024 FRC field:
#                        apriltags = [tag for tag in apriltags if ((tag.getId() >= 1) & (tag.getId() <= 16))]

                    tag_state = []
                    tag_state = ["LOST" for i in range(16)]
                    for tag in apriltags:
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
                        cv2.line(outFrame, corner0, corner1, color = col_box, thickness = 1)
                        cv2.line(outFrame, corner1, corner2, color = col_box, thickness = 1)
                        cv2.line(outFrame, corner2, corner3, color = col_box, thickness = 1)
                        cv2.line(outFrame, corner3, corner0, color = col_box, thickness = 1)

                        # Label the tag with the ID:
                        cv2.putText(outFrame, f"{tag_id}", (int(center.x), int(center.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, col_txt, 2)

                        # Update Tag data in networktables
                        if ((tag_id >= 1) & (tag_id <= 16)):
                            tag_state[tag_id-1] = "TRACKED";
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
                    latency2 = dai.Clock.now() - inGray.getTimestamp()
                    ssd=sd.getSubTable("FrontCam/Latency")
                    ssd.putNumber("Image", float(latency1.total_seconds()))
                    ssd.putNumber("Apriltag", float(latency2.total_seconds()))

                    cv2.putText(outFrame, "NN fps: {:.2f}".format(fps), (2, outFrame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

                    # After all the drawing is finished, we show the frame on the screen
                    # we can't do that on a headless RPi....           cv2.imshow("preview", frame)
                    # Instead publish to CameraServer output stream for NetworkTables or MJPEG http stream\
                    # ... and lower the refresh rate to comply with FRC robot wireless bandwidth regulations
                    image_output_bandwidth_limit_counter += 1
                    if image_output_bandwidth_limit_counter > 1:
                        image_output_bandwidth_limit_counter = 0
                        output_stream_nn.putFrame(outFrame)

                # print performance metrics
#                print(f"{latency1}; {latency2}")

            if cv2.waitKey(1) == ord('q'):
                break
