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
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

#width = camera['width']
#height = camera['height']
width = 640
height = 480

# Linking
monoLeft.out.link(xoutMono.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the mono frames from the outputs defined above
    monoQueue = device.getOutputQueue("mono", 8, False)

    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    # April Tag detection:
    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily("tag36h11")
    estimator = robotpy_apriltag.AprilTagPoseEstimator(
        robotpy_apriltag.AprilTagPoseEstimator.Config(
            0.2, 500, 500, width / 2.0, height / 2.0
        )
    )

    # Detect apriltag
    DETECTION_MARGIN_THRESHOLD = 100
    DETECTION_ITERATIONS = 50

    while(True):
        inFrame = monoQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        monoFrame = inFrame.getFrame()
        frame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)

        # Coordinates of found targets, for NT output:
        x_list = []
        y_list = []
        id_list = []

#        gray = cv2.cvtColor(monoFrame, cv2.COLOR_BGR2GRAY)
#        tag_info = detector.detect(gray)
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
            #decision_margin = tag.getDecisionMargin()
            print(f"{tag_id}: {pose}")

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

            x_list.append((center.x - width / 2) / (width / 2))
            y_list.append((center.y - width / 2) / (width / 2))
            id_list.append(tag_id)


#        aprilTagData = aprilTagQueue.get().aprilTags
#        for aprilTag in aprilTagData:
#            topLeft = aprilTag.topLeft
#            topRight = aprilTag.topRight
#            bottomRight = aprilTag.bottomRight
#            bottomLeft = aprilTag.bottomLeft

#            center = (int((topLeft.x + bottomRight.x) / 2), int((topLeft.y + bottomRight.y) / 2))

#            cv2.line(frame, (int(topLeft.x), int(topLeft.y)), (int(topRight.x), int(topRight.y)), color, 2, cv2.LINE_AA, 0)
#            cv2.line(frame, (int(topRight.x), int(topRight.y)), (int(bottomRight.x), int(bottomRight.y)), color, 2, cv2.LINE_AA, 0)
#            cv2.line(frame, (int(bottomRight.x), int(bottomRight.y)), (int(bottomLeft.x), int(bottomLeft.y)), color, 2, cv2.LINE_AA, 0)
#            cv2.line(frame, (int(bottomLeft.x), int(bottomLeft.y)), (int(topLeft.x), int(topLeft.y)), color, 2, cv2.LINE_AA, 0)

#            idStr = "ID: " + str(aprilTag.id)
#            cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.putText(frame, "Fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

        cv2.imshow("mono", frame)

        if cv2.waitKey(1) == ord('q'):
            break
