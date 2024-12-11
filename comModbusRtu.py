import threading
import time
import binascii

#from pymodbus.client.sync import ModbusSerialClient
from pymodbus.client.serial import ModbusSerialClient
from math import ceil

class ComModbusRtu:
    def __init__(self):
        """Constructor."""
        self.client = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_timeout = 0
        self._max_timeout = 255
        self._min_force = 0
        self._max_force = 255
    
    def connect(self, device):
        """Connection to the client - the method takes the IP address (as a string, e.g. 'COM14') as an argument."""
        self.client = ModbusSerialClient(method='rtu',port=device,stopbits=1, bytesize=8, baudrate=115200, timeout=0.2)
        if not self.client.connect():
            print("Unable to connect to %s" % device)
            return False
        return True

    def disconnect(self):
        """Close connection"""
        self.client.close()

    def reconnect(self):
        return self.connect()

    def sendCommand(self, data):   
        """Send a command to the Gripper - the method takes a list of uint8 as an argument. The meaning of each variable depends on the Gripper model (see support.robotiq.com for more details)"""
        #make sure data has an even number of elements 
         
        if(len(data) % 2 == 1):
            data.append(0)

        #Initiate message as an empty list
        message = []
        #Fill message by combining two bytes in one register
        for i in range(0, len(data)//2):
            message.append((data[2*i] << 8) + data[2*i+1])

        #To do!: Implement try/except 
        self.client.write_registers(0x03E8, message, unit=0x0009)

    def getStatus(self, numBytes):
        """Sends a request to read, wait for the response and returns the Gripper status. The method gets the number of bytes to read as an argument"""
        numRegs = int(ceil(numBytes/2.0))

        #To do!: Implement try/except 
        #Get status from the device
        response = self.client.read_holding_registers(0x07D0, numRegs, unit=0x0009)

        #Instantiate output as an empty list
        output = []

        #Fill the output with the bytes in the appropriate order
        for i in range(0, numRegs):
            output.append((response.getRegister(i) & 0xFF00) >> 8)
            output.append( response.getRegister(i) & 0x00FF)
        
        #Output the result
        return output