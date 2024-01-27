#!/usr/bin/env python3

import cv2
import depthai as dai
import time

import robotpy_apriltag
from wpimath.geometry import Transform3d
import math

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
xoutMono = pipeline.create(dai.node.XLinkOut)

xoutMono.setStreamName("mono")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(24)         # 30 is about the maximum on USB.SUPERSPEED
xoutMono.setStreamName("mono")

print('Resolution', monoLeft.getResolution())
print('Resolution', monoLeft.getResolutionSize())

width = monoLeft.getResolutionWidth()
height = monoLeft.getResolutionHeight()

# Linking
monoLeft.out.link(xoutMono.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Print MxID, USB speed, and available cameras on the device
    print('MxId:',device.getDeviceInfo().getMxId())
    print('USB speed:',device.getUsbSpeed())
    print('Connected cameras:',device.getConnectedCameras())

    # Output queue will be used to get the mono frames from the outputs defined above
    # use non-blocking queue access
    monoQueue = device.getOutputQueue("mono", 1, False)

    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

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
            0.165, 450, 450, width / 2.0, height / 2.0
        )
    )

    # Detect apriltag
    DETECTION_MARGIN_THRESHOLD = 100
    DETECTION_ITERATIONS = 50

    while(True):
        inFrame = monoQueue.tryGet()
        if inFrame:
            latency1 = dai.Clock.now() - inFrame.getTimestamp()

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            monoFrame = inFrame.getFrame()
            frame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)

            tag_info = detector.detect(monoFrame)
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
                latency2 = dai.Clock.now() - inFrame.getTimestamp()
                print(f"{latency1}; {latency2}")

            cv2.putText(frame, "Fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

            cv2.imshow("mono", frame)

        if cv2.waitKey(1) == ord('q'):
            break
