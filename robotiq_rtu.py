from typing import Any
from comModbusRtu import ComModbusRtu
import os, sys
import time


class RobotiqRtu:
    def __init__(self):
        self.client = ComModbusRtu()
    
    def connect(self,device):
        self.client.connect(device)

    def __del__(self):
        self.client.disconnect()

    def disconnect(self):
        self.client.disconnect()
    
    def reconnect(self):
        self.client.reconnect()

    def verifyCommand(self, command):
        """Function to verify that the value of each variable satisfy its limits."""
    	   	
        #Verify that each variable is in its correct range
        command.rACT = max(0, command.rACT)
        command.rACT = min(1, command.rACT)
        
        command.rMOD = max(0, command.rMOD)
        command.rMOD = min(1, command.rMOD)

        command.rGTO = max(0, command.rGTO)
        command.rGTO = min(1, command.rGTO)

        command.rATR = max(0, command.rATR)
        command.rATR = min(1, command.rATR)
        
        command.rPR  = max(0,   command.rPR)
        command.rPR  = min(255, command.rPR)   	

        command.rSP  = max(0,   command.rSP)
        command.rSP  = min(255, command.rSP)   	

        command.rFR  = max(0,   command.rFR)
        command.rFR  = min(255, command.rFR) 
        
        #Return the modified command
        return command

    def sendCommand(self):
        #Limit the value of each variable
        self.command = self.verifyCommand(self.command)
        
        #Initiate command as an empty list
        self.message = []

        #Build the command with each output variable
        #To-Do: add verification that all variables are in their authorized range
        self.message.append(self.command.rACT + (self.command.rMOD << 1) + (self.command.rGTO << 3) + (self.command.rATR << 4))
        self.message.append(0)
        self.message.append(0)
        self.message.append(self.command.rPR)
        self.message.append(self.command.rSP)
        self.message.append(self.command.rFR)    
        self.client.sendCommand(self.message)

    def getStatus(self):
        """Request the status from the gripper and return it in the RobotiqVacuumGrippers_robot_input msg type."""
        try:
            #Acquire status from the Gripper
            status = self.client.getStatus(6)

            #Message to output
            message = RtuStatus()

            #Assign the values to their respective variables
            message.gACT = (status[0] >> 0) & 0x01
            message.gMOD = (status[0] >> 1) & 0x03        
            message.gGTO = (status[0] >> 3) & 0x01
            message.gSTA = (status[0] >> 4) & 0x03
            message.gOBJ = (status[0] >> 6) & 0x03
            message.gFLT =  status[2]
            message.gPR  =  status[3]
            message.gPO  =  status[4]
        
        except:
            time.sleep(1)	 #Small delay for in case of synchronization issues	
            #Acquire status from the Gripper
            status = self.client.getStatus(6)

            #Message to output
            message = RtuStatus()

            #Assign the values to their respective variables
            message.gACT = (status[0] >> 0) & 0x01
            message.gMOD = (status[0] >> 1) & 0x03        
            message.gGTO = (status[0] >> 3) & 0x01
            message.gSTA = (status[0] >> 4) & 0x03
            message.gOBJ = (status[0] >> 6) & 0x03
            message.gVAS =  status[1] >> 0
            message.gFLT =  status[2]
            message.gPR  =  status[3]
            message.gPO  =  status[4]
        return message
    
    def showStatus(self):
        print(self.getStatus().show())

    def activate(self):
        self.command = RtuCommand()
        self.command.rACT = 1
        self.command.rMOD = 1
        self.command.rGTO = 1
        self.command.rPR = 255
        self.command.rSP  = 255
        self.command.rFR  = 50

        return self.sendCommand()
    
    def grip_on(self, pressure=0):
        self.command.rPR = pressure
        self.sendCommand()   

    def grip_off(self):
        self.command.rPR = 255
        self.sendCommand()

    def getPressure(self):
        status = self.getStatus()
        return status.gPO

# Reference:https://assets.robotiq.com/website-assets/support_documents/document/EPick_Instruction_Manual_CB-Series_PDF_20200420.pdf
class RtuCommand:
    def __init__(self):
        self.rACT = 0 # Activate
        self.rMOD = 0 #  Change mode, 0: AUTOMATIC, 1:ADVANCED
        self.rGTO = 0 # REGULATE, 0:
        self.rATR = 0 # AUTOMATIC RELEASE
        self.rPR  = 0 # Pressure request, ONLY VALID IN MANUAL MODE
        self.rSP  = 0 # TIMEOUT, 0:NO LIMIT, 1:1 SEC TIMEOUT PERIOD, 255:25.5 SEC TIMEOUT PERIOD
        self.rFR  = 0 # Minimum vacuum
    
    def show(self):
        # currentCommand  = '\nSimple vacuum Gripper Controller\n-----\nCurrent command:'
        currentCommand = ""
        currentCommand += '  rACT = '  + str(self.rACT)
        currentCommand += '  rMOD = '  + str(self.rMOD)
        currentCommand += ', rGTO = '  + str(self.rGTO)
        currentCommand += ', rATR = '  + str(self.rATR)
        currentCommand += ', rPR = '   + str(self.rPR )
        currentCommand += ', rSP = '   + str(self.rSP )
        currentCommand += ', rFR = '   + str(self.rFR )
        # currentCommand += '\n-----'
        return(currentCommand)

class RtuStatus:
    def __init__(self):
        self.gACT = 0 # Activate echo
        self.gMOD = 0 # Grippermode echo
        self.gGTO = 0 # Regulate echo
        self.gSTA = 0 # Activationstatus, 0:Gripper is not activated, 1:Gripper is operational
        self.gOBJ = 0 # Object status, 0:Unknown object detection, 1:Object detected. Minimum vacuum value reached., 2:Object detected. Maxinum vacuum value reached., 3:No object detected. Object loss, dropped or gripping timeout reached.
        self.gVAS = 0 # Vacuum actuator status
        self.gFLT = 0 # Gripper fault status
        self.gPR  = 0 # Vacuum/Pressure request echo
        self.gPO  = 0 # Actual Vacuum/Pressure
    
    def show(self):
        # currentStatus  = '\nSimple vacuum Gripper Controller\n-----\nCurrent status:'
        currentStatus = ""
        currentStatus += '  gACT = '  + str(self.gACT)
        currentStatus += '  gMOD = '  + str(self.gMOD)
        currentStatus += ', gGTO = '  + str(self.gGTO)
        currentStatus += ', gSTA = '  + str(self.gSTA)
        currentStatus += ', gOBJ = '  + str(self.gOBJ)
        currentStatus += ', gVAS = '  + str(self.gVAS)
        currentStatus += ', gFLT = '  + str(self.gFLT)
        currentStatus += ', gPR = '   + str(self.gPR )
        currentStatus += ', gPO = '   + str(self.gPO )
        # currentStatus += '\n-----'
        return(currentStatus)
    

if __name__ == "__main__":
    gripper = RobotiqRtu()
    gripper.connect("COM3")
    gripper.activate()

    s_time = time.time()
    e_time = time.time()-s_time
    gripper.grip_on(pressure=0)
    print("grip on")
    while e_time < 10:
        e_time = time.time()-s_time
        # print(e_time)

        gripper.showStatus()
    gripper.grip_off()
    print("grip off")
    print(gripper.getPressure())

    gripper.disconnect()