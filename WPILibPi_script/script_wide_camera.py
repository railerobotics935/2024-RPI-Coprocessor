# Script based off of the depthAI versions for solely the mono wide camera (non depthAI)
# Team 935
#
# From Jaap
# DONE: Setup Pi configuration stuff
# TODO: Setup NT Communication
#
# With Guidence
# TODO: Setup Apriltag processing
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

# From Jaap
# This will be used to set up the pi (Read the config file)
# =============================================
from cscore import CameraServer
from ntcore import NetworkTableInstance

# For AprilTag processing
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
