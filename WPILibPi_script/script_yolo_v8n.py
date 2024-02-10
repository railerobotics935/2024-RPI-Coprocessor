#!/usr/bin/env python3
#
# based on: https://docs.luxonis.com/projects/api/en/v2.1.0.0/samples/26_1_spatial_mobilenet/
# and: https://docs.luxonis.com/projects/api/en/latest/samples/ObjectTracker/spatial_object_tracker/#spatial-object-tracker-on-rgb
# updated to work on FRC 2024 WPILibPi image and to be uploaded as a vision application
# communicating with shuffleboard and RoboRIO through NetworkTables and CameraServer
# Jaap van Bergeijk, 2024

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

    # Label texts
    labelMap = ["robot", "note"]
    syncNN = True
    wideFOV = True
    stereoDepth = True

    # Static setting for model BLOB, this runs on a RPi with a RO filesystem
    nnPath = str((Path(__file__).parent / Path('models/YOLOv8nNORO_openvino_2022.1_6shave416.blob')).resolve().absolute())

    # Create a CameraServer for ShuffleBoard visualization
    CameraServer.enableLogging()
#    camera = CameraServer.startAutomaticCapture()
#    cs.enableLogging()

    # Width and Height have to match Neural Network settings: 416x416 pixels
    nn_width = 416
    nn_height = 416
    output_stream_nn = CameraServer.putVideo("FrontNN", nn_width, nn_height)

    # Width and Height of various image processing pipelines (optimized for speed and bandwidth)
    rgb_preview_width = 812
    rgb_preview_height = 608

    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = dai.Pipeline()

    # First, we want the Color camera as the output
    camRgb = pipeline.create(dai.node.ColorCamera)
    if wideFOV:
        manip = pipeline.create(dai.node.ImageManip)
    if stereoDepth:
        detectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
    else:
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    objectTracker = pipeline.createObjectTracker()

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutTracker = pipeline.create(dai.node.XLinkOut)
    xoutPreview = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutTracker.setStreamName("tracklets")
    xoutPreview.setStreamName("preview")
       
    # Properties
    if wideFOV:
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        camRgb.setInterleaved(False)
#        camRgb.setIspScale(1,5) # 4056x3040 -> 812x608
#        camRgb.setPreviewSize(812, 608)
        camRgb.setIspScale(1,4) # 4056x3040 -> 1014x760
        camRgb.setPreviewSize(1014, 760)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(20)
    else:
        camRgb.setPreviewSize(320, 320)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)

    # Network specific settings
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(2)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors([])
    detectionNetwork.setAnchorMasks({})
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)
#    detectionNetwork.setDepthLowerThreshold(100)
#    detectionNetwork.setDepthUpperThreshold(15000)

    # Use ImageManip to resize to 416x416 with letterboxing: enables a wider FOV
    if wideFOV:
        manip.setMaxOutputFrameSize(int(nn_width * nn_height * 3))
        manip.initialConfig.setResizeThumbnail(nn_width, nn_height)
        camRgb.preview.link(manip.inputImage)

    # setting node configs
    if stereoDepth:
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

    objectTracker.setDetectionLabelsToTrack([0, 1])  # track robots and notes
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    # Linking
    if stereoDepth:
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

    if wideFOV:
        manip.out.link(detectionNetwork.input)
    else:
        camRgb.preview.link(detectionNetwork.input)

    # BEGIN insert for object tracking test
    objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
    objectTracker.out.link(xoutTracker.input)

    detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)
    stereo.depth.link(detectionNetwork.inputDepth)
    
#    camRgb.preview.link(xoutPreview.input)     # This did NOT work on the RPi, not enough USB bandwidth for this higher res output stream
    
    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    #with dai.Device(pipeline, True) as device:
    with dai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink

        # Output queues will be used to get the rgb frames, tracklets data and depth frames from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qTracklets = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        image_output_bandwidth_limit_counter = 0

        while True:
#            inRgb = qRgb.get()
            inRgb = qRgb.tryGet()
#            track = qTracklets.get()
            track = qTracklets.tryGet()

            if inRgb is not None:
                latency1 = dai.Clock.now() - inRgb.getTimestamp()
                frame = inRgb.getCvFrame()

#                print(f"{frame.shape[1]} : {frame.shape[0]}") # should be 416 x 416 (or nn_width x nn_height)
#                print(f"{depthFrame.shape[1]} : {depthFrame.shape[0]}") # should be 640 x 400

                if frame is not None:
                    counter+=1
                    current_time = time.monotonic()
                    if (current_time - startTime) > 1:
                        fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time

                    if track is not None:
                        trackletsData = track.tracklets

                        # If the frame is available, draw bounding boxes on it and show the frame
                        for t in trackletsData:
                            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                            x1 = int(roi.topLeft().x)
                            y1 = int(roi.topLeft().y)
                            x2 = int(roi.bottomRight().x)
                            y2 = int(roi.bottomRight().y)

                            try:
                                label = labelMap[t.label]
                            except:
                                label = t.label

                            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                            cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                            cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                            cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                            cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)

                            ssd=sd.getSubTable(f"OakDLite/Object[{t.id}]")
                            ssd.putString("Label", str(label))
                            ssd.putString("Status", t.status.name)
                #            sd.putNumber("Confidence", int(detection.confidence * 100))
                            ssd.putNumberArray("Location", [int(t.spatialCoordinates.x), int(t.spatialCoordinates.y), int(t.spatialCoordinates.z)])

                    # Update Latency data in networktables
                    latency2 = dai.Clock.now() - inRgb.getTimestamp()
                    ssd=sd.getSubTable("OakDLite/Latency")
                    ssd.putNumber("Image", float(latency1.total_seconds()))
                    ssd.putNumber("ObjectDetect", float(latency2.total_seconds()))

                    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

                    # After all the drawing is finished, we show the frame on the screen
                    # we can't do that on a headless RPi....           cv2.imshow("preview", frame)
                    # Instead publish to CameraServer output stream for NetworkTables or MJPEG http stream\
                    # ... and lower the refresh rate to comply with FRC robot wireless bandwidth regulations
                    image_output_bandwidth_limit_counter += 1
                    if image_output_bandwidth_limit_counter > 1:
                        image_output_bandwidth_limit_counter = 0
                        output_stream_nn.putFrame(frame)

            if cv2.waitKey(1) == ord('q'):
                break
