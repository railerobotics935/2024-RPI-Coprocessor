This repository contains the information and code for using a RPi coprocessor for camera stream processing

The functions supported are:

- Using DepthAI with an OAK-D Lite camera for object recognition and object location
- Using the OAK-D Lite camera with the robotpy-apriltag library to detect 36H11 encoded april tags

Preparation of a RPi

1) write image to 16 Gb uSD card, use 64 bit image (WPILibPi_64_image-v2023.2.1.zip)
      https://github.com/wpilibsuite/WPILibPi/releases
2) power up RPi, connect ethernet cable to a network with access to internet
3) open up a browser, navigate to wpilibpi.local
4) make file system writable (webbrowser to wpilibpi.local)
5) open ssh session with wpilibpi.local (e.g. with PuTTY application, log in with username "pi", password "raspberry")
6) sudo date -s 'fri mar 3 12:07:23 CST 2023'  (has to be exact!)

  install depthai
      https://docs.luxonis.com/projects/api/en/latest/install/
      https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os
      https://docs.luxonis.com/projects/api/en/latest/install/#install-from-pypi
7) sudo apt-get install python3-venv
8) sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
9) python3 -m pip install depthai

  install robotpy-apriltag
      https://pypi.org/project/robotpy-apriltag/
10) pip install robotpy-apriltag (was already installed, just to check)
11) open winscp session with wpilibpi.local
12) create folder "models" under /home/pi
13) copy neural network models to the models folder
14) upload application and configure team number, camera, etc in browser wpilibpi.local interface
15) connect Shuffleboard either directly to the RPi (wpilibpi.local) or through radio to RoboRIO (10.9.35.2)
