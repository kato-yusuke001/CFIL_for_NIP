import sys
sys.dont_write_bytecode = True # キャッシュを生成させない
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
from dashboard_client import DashboardClient

import time

# Connect to the UR robot
# ROBOT_IP = "192.168.56.2"
ROBOT_IP = "192.168.11.77"
rtde_c = None
rtde_r = None
io = None
dashboard = None

def connect():
    global rtde_c, rtde_r, gripper, dashboard, io
    dashboard = DashboardClient(ROBOT_IP)
    dashboard.connect()
    # dashboard.powerOff()
    # dashboard.restartSafety()
    time.sleep(3)
    print(dashboard.safetystatus(), dashboard.safetymode(), dashboard.robotmode(), dashboard.running())
    if(dashboard.isConnected() and dashboard.safetystatus() == "Safetystatus: NORMAL" and dashboard.robotmode() == "Robotmode: POWER_OFF"):
        print("robot wakeup")
        # dashboard.restartSafety()
        dashboard.powerOn()
        dashboard.brakeRelease()
        dashboard.unlockProtectiveStop()
        s_time = time.time()
        while(dashboard.robotmode() != "Robotmode: RUNNING"):
            if(time.time()-s_time > 10):
                print("Failed to wakeup robot")
                quit()
            print(dashboard.robotmode()) 
            time.sleep(1)
            
    print("dashboard connected")
    dashboard.closePopup()
    s_time = time.time()
    # while(not dashboard.running()):
    #     if(time.time()-s_time > 30):
    #         print("Failed to wakeup robot")
    #         quit()
    #     print(dashboard.running())
    #     time.sleep(1)
    # print(dashboard.safetystatus(), dashboard.safetymode())
    try:
        print(dashboard.safetystatus(), dashboard.safetymode(), dashboard.robotmode(), dashboard.running())
        if(rtde_c is None): rtde_c = RTDEControlInterface(ROBOT_IP)
        print("rtde_c connected")
        if(rtde_r is None): rtde_r = RTDEReceiveInterface(ROBOT_IP)
        print("rtde_r connected")
        if(io is None): io = RTDEIOInterface(ROBOT_IP)
        print("io connected")
        print(dashboard.safetystatus(), dashboard.safetymode(), dashboard.robotmode(), dashboard.running())
    except Exception as e:
        print(type(e), e)
        print("Failed to connect robot")
    
def reconnect():
    global rtde_c, rtde_r, dashboard, io
    try:
        if rtde_c is not None:
            rtde_c.disconnect()
            rtde_c = None
        if rtde_r is not None:
            rtde_r.disconnect()
            rtde_r = None
        if io is not None:
            io.disconnect()
            io = None

        rtde_r = RTDEReceiveInterface(ROBOT_IP)
        rtde_c = RTDEControlInterface(ROBOT_IP)
    
        io = RTDEIOInterface(ROBOT_IP)
    
        print("Robot ReConnection!")
    except Exception as e:
        print(type(e), e)

def wakeup():
    global rtde_c, rtde_r, dashboard, io
    if rtde_c is None or rtde_r is None or dashboard is None or io is None: 
        print(" robot is not connected")
        return
    try:
        print("Safety Mode", rtde_r.getSafetyMode(), dashboard.safetymode())
        print("Robot Status", rtde_r.getRobotStatus(), dashboard.safetystatus())
        if(rtde_r.getRobotStatus()==3):
            dashboard.closeSafetyPopup()
            print(" Status Normal")
            return
        print("wakeup robot ...")
        if(dashboard.isConnected()):
            dashboard.closeSafetyPopup()
            
            if(rtde_r.getSafetyMode() == 9):
                print(" Status Failed", rtde_r.getSafetyMode())
                dashboard.restartSafety()
                time.sleep(3)
                dashboard.powerOn()
                time.sleep(3)
                dashboard.brakeRelease()
                dashboard.unlockProtectiveStop()
                rtde_c.disconnect()
                rtde_r.disconnect()
                # io.disconnect()
                rtde_c.reconnect()
                rtde_r.reconnect()
                # io.reconnect()
            elif(rtde_r.getSafetyMode() == 7):
                print(" Status Emergency Button still pressed", rtde_r.getSafetyMode())
                return
            else:
                print(" Status Standby", rtde_r.getSafetyMode())
                dashboard.powerOn()
                dashboard.brakeRelease()
                dashboard.unlockProtectiveStop()
                print("  Unlock Protective Stop")
                io.setConfigurableDigitalOut(0,1)
                io.setConfigurableDigitalOut(1,1)
                print("  io reset")
                s_time = time.time()
                while(dashboard.robotmode() != "Robotmode: RUNNING"):
                    if(time.time()-s_time > 10):
                        print("Failed to wakeup robot")
                        quit()
                    print(dashboard.robotmode()) 
                    time.sleep(1)

                # rtde_c.disconnect()
                # rtde_r.disconnect()
                # time.sleep(3)
                # print("   disconnect")
                # rtde_r = RTDEReceiveInterface(ROBOT_IP)
                # rtde_c = RTDEControlInterface(ROBOT_IP)
                # print('   Robot Reconnected')
                
                print("Robot status", rtde_r.getRobotStatus(),dashboard.robotmode())
            s_time = time.time()
            # while(rtde_r.getRobotStatus()!=3):
            #     if(time.time()-s_time > 30):
            #         print("Failed to wakeup robot")
            #         return
            #     print(rtde_r.getRobotStatus(), rtde_r.getSafetyMode())
            #     rtde_c.disconnect()
            #     rtde_r.disconnect()
            #     rtde_c.reconnect()
            #     rtde_r.reconnect()
            print("Completed to wakeup robot")
            return
    except Exception as e:
        print(type(e), e)
        return
    
def disconnect():
    global rtde_c, rtde_r, dashboard, io
    if rtde_c is not None:
        rtde_c.disconnect()
        rtde_c = None
        print("rtde_c disconnected")
    if rtde_r is not None:
        rtde_r.disconnect()
        rtde_r = None 
        print("rtde_r disconnected")
    if io is not None:
        io.disconnect()
        io = None
        print("io disconnected")
    if dashboard is not None:
        dashboard.disconnect()
        dashboard = None
        print("dashboard disconnected")
    
connect()

dashboard.powerOff()
time.sleep(3)
wakeup()


# disconnect()
# time.sleep(1)
# wakeup()
# time.sleep(1)
# print(rtde_r.getActualTCPPose())
# time.sleep(1)
# reconnect()
# time.sleep(1)
# print(rtde_r.getActualTCPPose())
