# Script based off of the depthAI versions for solely the mono wide camera (non depthAI)
# Team 935

# Calculates the 2d translation of the note based off of object size and position of the object

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

# global constants
configFile = "/boot/frc.json"
team = 935
server = False

# Label texts
labelMap = ["robot", "note"]
syncNN = True
wideFOV = True
stereoDepth = True


DETECTION_MARGIN_THRESHOLD = 50
DETECTION_ITERATIONS = 50

cam1Name = "FrontCam"
cam2Name = "BackLeftCam"
cam3Name = "BackRightCam"
#    cam4Name = "OakDLite"

# Width and Height of various image processing pipelines (optimized for speed and bandwidth)
# USB OV9282 camera -> 1280x800 pixels
# OAK-D Lite mono camera -> 640x400 pixels
ov9282_width = 1280
ov9282_height = 800
odlite_gray_width = 640
odlite_gray_height = 400


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


# ==========================================================================================================================================================
# initialize the AprilTag processing
def configureAprilTagDetection():

    # Create the Apriltag detector, common for all types of cameras
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

    # Create the Apriltag Pose estimators, need 2 due to different image dimensions
    ov9282Estimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            0.165, 730, 730, ov9282_width / 2.0, ov9282_height / 2.0
        )
    )

    return detector, ov9282Estimator


# ==========================================================================================================================================================
# detect Apriltags in a frame, Estimate Pose if any are found and update Network Tables
def processApriltags(gray_frame, nt_name, detector, estimator):

    tag_state = []
    tag_state = ["LOST" for i in range(16)]

    # Feed gray scale image into AprilTag library
    tag_info = detector.detect(gray_frame)
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

    # Update Tag data in networktables
    for i in range(16):
        if tag_state[i] == "LOST" :
            ssd=sd.getSubTable(nt_name + f"/Tag[{i+1}]")
            if ssd.getString("Status", "TRACKED") != "LOST" :
                ssd.putString("Status", "LOST")
                ssd.putNumberArray("Pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# ==========================================================================================================================================================
# capture an AprilTag detection image from a USB video Capture
# pass Network Tables entry string and Apriltag Pose Estimator on to Apriltag processor
def processOV9282Apriltags(cap, nt_name, detector, estimator):
#    print("processing " + nt_name)
    hasData, frame = cap.read()
    inputImageTime = time.monotonic()
    if (hasData):
#        print("processing has data " + nt_name)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processApriltags(gray, nt_name, detector, estimator)
    else:
        # this detects a camera malfunction, only reset tag state to
        # "LOST" when this state is reached for longer than the image update rate
#       print("processing has NO data " + nt_name)
        # Set data to lost and zero
        tag_state = []
        tag_state = ["LOST" for i in range(16)]

        # Update Tag data in networktables
        for i in range(16):
            if tag_state[i] == "LOST" :
                ssd=sd.getSubTable(nt_name + f"/Tag[{i+1}]")
                if ssd.getString("Status", "TRACKED") != "LOST" :
                    ssd.putString("Status", "LOST")
                    ssd.putNumberArray("Pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Update Latency data in networktables
    latency2 = time.monotonic() - inputImageTime
    ssd=sd.getSubTable(nt_name + "/Latency")
    ssd.putNumber("Apriltag", float(latency2))
#    ntinst.flush()   disabled here, flush all 3 cams at the end of a data processing cycle


# =============================================================================
# capture an AprilTag detection image from a OAK-D Lite video Capture
# pass Network Tables entry string and Apriltag Pose Estimator on to Apriltag processor
def processODLiteApriltags(qCam, nt_name, estimator):

    inGray = qCam.tryGet()

    if inGray is not None:
#        latency1 = dai.Clock.now() - inGray.getTimestamp()

        gray = inGray.getFrame()
        if gray is not None:
            processApriltags(gray, nt_name, estimator)

    # Update Latency data in networktables
    latency2 = dai.Clock.now() - inGray.getTimestamp()
    ssd=sd.getSubTable(nt_name + "/Latency")
    ssd.putNumber("Apriltag", float(latency2.total_seconds()))
#    ntinst.flush()


# =============================================================================
# initialize the OAK-D Lite AprilTag processing pipeline
def configureODLiteAprilTagDetectionPipeline():

    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = dai.Pipeline()

    # Use the Left Mono camera as the output, this is a monochrome camera, add input and output nodes to the pipeline
    monoLeft = pipeline.create(dai.node.MonoCamera)
    xoutGray = pipeline.create(dai.node.XLinkOut)

    # Set the Properties of the Pipeline nodes
    xoutGray.setStreamName("gray")
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoLeft.setFps(20)

    # Linking the Pipeline nodes
    monoLeft.out.link(xoutGray.input)

    # Create the Apriltag detector, common for all types of cameras
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

    # Create the Apriltag Pose estimators, need 2 due to different image dimensions
    ov9282Estimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            0.165, 730, 730, ov9282_width / 2.0, ov9282_height / 2.0
        )
    )

    odliteEstimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            0.165, 470, 470, odlite_gray_width / 2.0, odlite_gray_height / 2.0
        )
    )
    
    # Detect apriltag
    DETECTION_MARGIN_THRESHOLD = 50
    DETECTION_ITERATIONS = 50

    return pipeline


# ==========================================================================================================================================================
# initialize the OAK-D Lite neural network processing pipeline
def configureODLiteObjectDetectionPipeline():

    # Static setting for model BLOB, this runs on a RPi with a RO filesystem
    nnPath = str((Path(__file__).parent / Path('models/YOLOv5nNORO_openvino_2022.1_6shave416.blob')).resolve().absolute())

    # nn_size has to match Neural Network settings: 416 pixels square
    nn_size = 416
    output_stream = CameraServer.putVideo("ObjectCam", nn_size, nn_size)

    # Width and Height of various image processing pipelines (optimized for speed and bandwidth)
    # use an image manipulation prescaler to scale down 1/4 from the color camera: 4056x3040 -> 1014x760
    isp_num = 1
    isp_den = 4
    rgb_width = 1014
    rgb_height = 760

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

    xoutRgb.setStreamName("rgb")
    xoutTracker.setStreamName("tracklets")
       
    # Properties
    if wideFOV:
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        camRgb.setInterleaved(False)
        camRgb.setIspScale(isp_num, isp_den)
        camRgb.setPreviewSize(rgb_width, rgb_height)
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
        manip.setMaxOutputFrameSize(int(nn_size * nn_size * 3))
        manip.initialConfig.setResizeThumbnail(nn_size, nn_size)
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

    return pipeline, output_stream


# ==========================================================================================================================================================
def processODLiteObjects(qRgb, qTracklets, image_output_bandwidth_limit_counter, output_stream_nn):

    inRgb = qRgb.tryGet()
    track = qTracklets.tryGet()
    color = (255, 255, 255)

    if inRgb is not None:
        latency1 = dai.Clock.now() - inRgb.getTimestamp()
        frame = inRgb.getCvFrame()

#       print(f"{frame.shape[1]} : {frame.shape[0]}") # should be 416 x 416 (or nn_width x nn_height)

        if frame is not None:
#            counter+=1
#            current_time = time.monotonic()
#            if (current_time - startTime) > 1:
#                fps = counter / (current_time - startTime)
#                counter = 0
#                startTime = current_time

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
#
                    cv2.putText(frame, "X: {:.3f} m".format(float(t.spatialCoordinates.x)*0.001), (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                    cv2.putText(frame, "Y: {:.3f} m".format(float(-t.spatialCoordinates.y)*0.001), (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                    cv2.putText(frame, "Z: {:.3f} m".format(float(t.spatialCoordinates.z)*0.001), (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)

                    # if the object is a note, do special processing
                    if (str(label) == "note"):
                        #if the object is a good note, allow new processing, else, lable as bad
                        if (validNote((x2-x1), (y2-y1))):
                            translation = getNote2dTranslation(x1, x2)
                            ssd=sd.getSubTable(f"ObjectCam/Object[{t.id}]")
                            ssd.putString("Label", str(label))
                            ssd.putString("Status", t.status.name)
                #            sd.putNumber("Confidence", int(detection.confidence * 100))
                            ssd.putNumberArray("Pose", [translation[0], translation[1], 0.0])
                        else:
                            ssd=sd.getSubTable(f"ObjectCam/Object[{t.id}]")
                            ssd.putString("Label", str(label))
                            ssd.putString("Status", "BAD")
                #            sd.putNumber("Confidence", int(detection.confidence * 100))
                            
                            # Rearange to be in NWU
                            ssd.putNumberArray("Pose", [float(t.spatialCoordinates.z)*0.001, -float(t.spatialCoordinates.x)*0.001, -float(t.spatialCoordinates.y)*0.001])
                    else:
                        # If not note, do normal processing
                        ssd=sd.getSubTable(f"ObjectCam/Object[{t.id}]")
                        ssd.putString("Label", str(label))
                        ssd.putString("Status", t.status.name)
            #            sd.putNumber("Confidence", int(detection.confidence * 100))
                        
                        # Rearange to be in NWU
                        ssd.putNumberArray("Pose", [float(t.spatialCoordinates.z)*0.001, -float(t.spatialCoordinates.x)*0.001, -float(t.spatialCoordinates.y)*0.001])

            # Update Latency data in networktables
            latency2 = dai.Clock.now() - inRgb.getTimestamp()
            ssd=sd.getSubTable("ObjectCam/Latency")
            ssd.putNumber("Image", float(latency1.total_seconds()))
            ssd.putNumber("ObjectDetect", float(latency2.total_seconds()))

#            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

            # After all the drawing is finished, we show the frame on the screen
            # we can't do that on a headless RPi....           cv2.imshow("preview", frame)
            # Instead publish to CameraServer output stream for NetworkTables or MJPEG http stream\
            # ... and lower the refresh rate to comply with FRC robot wireless bandwidth regulations
            image_output_bandwidth_limit_counter += 1
            if image_output_bandwidth_limit_counter > 2:
                image_output_bandwidth_limit_counter = 0
                #output_stream_nn.putFrame(frame)

    return image_output_bandwidth_limit_counter


# ==========================================================================================================================================================
# returns true if the ratio between the width and height is within an expected value 
# this is to rule out notes that are sitting next to each other, or if there is an unfortuate orange robot
def validNote(width, height):
    if (width / height > 1.5 and width / height < 5.5): # based on data
        return True
    else:
        return False


# ==========================================================================================================================================================
# Does math to determine the distance of the note from the camera
# Experementaly gathered, returns in meters
def getNoteDistance(width):
    return 139 * pow(width, -1.16)


# ==========================================================================================================================================================
# Returns the percentage the note is horizontaly in the screen. left to right, 0.00 to 1.00 
def getNotePercentAcrossScreen(centerx):
    return (centerx / 320)


# ==========================================================================================================================================================
# Does math to deterime camera relative translation of the note in array
def getNote2dTranslation(x1, x2):
    centerx = x1 + (0.5 * (x2 - x1))
    width = x2 - x1
    
    noteDistance = getNoteDistance(width)
    noteTheta = (1.129) * (abs(0.5 - getNotePercentAcrossScreen(centerx))) # 1.129 is the measured fov after processing

    translationx = noteDistance * (math.sin(noteTheta))
    translationy = noteDistance * (math.cos(noteTheta))

    # put in the x and y translation relative to the camera
    noteTranslation = [translationx, translationy]

    return noteTranslation

# ==========================================================================================================================================================
# main application entry point
if __name__ == "__main__":
    time.sleep(5) # to make sure the rio is good

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

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    cap3 = cv2.VideoCapture(4)
#    print(f"{cap1}")
#    print(f"{cap2}")

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, ov9282_width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, ov9282_height)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, ov9282_width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, ov9282_height)

    cap3.set(cv2.CAP_PROP_FRAME_WIDTH, ov9282_width)
    cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, ov9282_height)

    # Configure the AprilTag detector
    detector, ov9282Estimator = configureAprilTagDetection()


    # DepthAI OAK-D Lite initialization
    pipeline, output_stream_nn = configureODLiteObjectDetectionPipeline()

    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    with dai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink
        image_output_bandwidth_limit_counter = 0

        # Output queues will be used to get the gray frames
        #qGray = device.getOutputQueue(name="gray", maxSize=1, blocking=False)

        # Output queues will be used to get the rgb frames, tracklets data and depth frames from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        qTracklets = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)

        # fps calculation
        fps = 0
        fps_counter = 0
        fps_reference_time = time.monotonic()

        # network tables flush pacing
        nt_last_flush_timestamp = time.monotonic() - 0.1

        while True:
            processOV9282Apriltags(cap1, cam1Name, detector, ov9282Estimator)
            processOV9282Apriltags(cap2, cam2Name, detector, ov9282Estimator)
            processOV9282Apriltags(cap3, cam3Name, detector, ov9282Estimator)
        #        processODLiteApriltags(qGray, cam4Name, odliteEstimator)
            image_output_bandwidth_limit_counter = processODLiteObjects(qRgb, qTracklets, image_output_bandwidth_limit_counter, output_stream_nn)

            fps_counter = fps_counter + 1
            fps_timestamp = time.monotonic()
            if (fps_timestamp - fps_reference_time) >= 1.0 :
                fps_reference_time = fps_timestamp
                fps = fps_counter
                fps_counter = 0
                ssd=sd.getSubTable("FrontCam")
                ssd.putNumber("FPS", float(fps))

            # flush all camera data at the end of a data processing cycle
            # minimum interval between flush commands is: 10 ms
            nt_flush_timestamp = time.monotonic()
            if (nt_flush_timestamp - nt_last_flush_timestamp) >= 0.01 :
                ntinst.flush()
                nt_last_flush_timestamp = nt_flush_timestamp

        #        disable this check, useless on RPi coprocessor
        #        if cv2.waitKey(1) == ord('q'):
        #            break

        # After the loop release the cap object
        cap1.release()
        cap2.release()
        cap3.release()
