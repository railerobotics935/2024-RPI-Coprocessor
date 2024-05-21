#!/usr/bin/env python3
#
# based on: https://docs.luxonis.com/projects/api/en/v2.1.0.0/samples/26_1_spatial_mobilenet/
# and: https://docs.luxonis.com/projects/api/en/latest/samples/ObjectTracker/spatial_object_tracker/#spatial-object-tracker-on-rgb
# updated to work on FRC 2024 WPILibPi image and to be uploaded as a vision application
# communicating with shuffleboard and RoboRIO through NetworkTables and CameraServer
# Jaap van Bergeijk, 2024


# Displays the width and height of the object in px


#from operator import truediv
from pathlib import Path

#for DepthAI processing
import json
import time
import sys
import cv2
import depthai as dai
import numpy as np

#for AprilTag processing
#import robotpy_apriltag
#from wpimath.geometry import Transform3d
#import math


# application entry point
if __name__ == "__main__":

    # Label texts
    labelMap = ["robot", "note"]
    syncNN = True
    wideFOV = True
    stereoDepth = True

    # Static setting for model BLOB
    nnPath = str((Path(__file__).parent / Path('YOLOv5nNORO_openvino_2022.1_4shave416.blob')).resolve().absolute())

    # nn_size has to match Neural Network settings: 416 pixels square
    nn_size = 416

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
    
    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    #with dai.Device(pipeline, True) as device:
    with dai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink
#        device.setLogLevel(dai.LogLevel.TRACE)
#        device.setLogOutputLevel(dai.LogLevel.TRACE)

        # Print MxID, USB speed, and available cameras on the device
        print('MxId:',device.getDeviceInfo().getMxId())
        print('USB speed:',device.getUsbSpeed())
        print('Connected cameras:',device.getConnectedCameras())

        # Output queues will be used to get the rgb frames, tracklets data and depth frames from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qTracklets = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)

        while True:
            inRgb = qRgb.tryGet()
            track = qTracklets.tryGet()

#            if track is not None:
#                track_latency = dai.Clock.now() - track.getTimestamp()
#                print(f"{track_latency}")

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

                            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
                            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
                            cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                            cv2.putText(frame, "W: {:.3f} px".format(x2 - x1), (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
                            cv2.putText(frame, "H: {:.3f} px".format(y2 - y1), (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
                            cv2.putText(frame, "X: {:.3f} m".format(float(t.spatialCoordinates.x)*0.001), (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
                            cv2.putText(frame, "Y: {:.3f} m".format(float(-t.spatialCoordinates.y)*0.001), (x1 + 10, y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)
                            cv2.putText(frame, "Z: {:.3f} m".format(float(t.spatialCoordinates.z)*0.001), (x1 + 10, y1 + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

                    # Update Latency data in networktables
                    latency2 = dai.Clock.now() - inRgb.getTimestamp()
#                    ssd=sd.getSubTable("OakDLite/Latency")
#                    ssd.putNumber("Image", float(latency1.total_seconds()))
#                    ssd.putNumber("ObjectDetect", float(latency2.total_seconds()))

#                    print(f"{latency1} : {latency2}")
                    cv2.putText(frame, "NN fps: {:.1f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

                    # After all the drawing is finished, we show the frame on the screen
                cv2.imshow("frame", frame)

            if cv2.waitKey(1) == ord('q'):
                break
