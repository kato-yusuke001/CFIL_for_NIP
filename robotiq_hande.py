"""Module to control Robotiq's epick"""

import socket
import threading
import time
from enum import Enum
from typing import Union
from collections import OrderedDict

# from torch import greater

class RobotiqHandE:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """
    # WRITE VARIABLES (CAN ALSO READ)
    ACT = 'ACT'  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = 'GTO'  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = 'ATR'  # atr : auto-release (emergency slow move)
    ADR = 'ADR'  # adr : auto-release direction (open(1) or close(0) during auto-release)
    FOR = 'FOR'  # for : force (0-255)
    SPE = 'SPE'  # spe : timeout (0-255)
    POS = 'POS'  # pos : position (0-255), 0 = grip on
    # READ VARIABLES
    STA = 'STA'  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = 'PRE'  # position request (echo of last commanded position)
    OBJ = 'OBJ'  # object detection (0 = moving, 1 = outer grip, 2 = inner grip, 3 = no object at rest)
    FLT = 'FLT'  # fault (0=ok, see manual for errors if not zero)

    ENCODING = 'UTF-8'  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""
        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        """Constructor."""
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_timeout = 0
        self._max_timeout = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        """Connects to a gripper at the given address.
        :param hostname: Hostname or ip.
        :param port: Port.
        :param socket_timeout: Timeout for blocking socket operations.
        """
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        self.socket.close()

    def _set_vars(self, var_dict: OrderedDict([(str, Union[int, float])])):
    # def _set_vars(self, var_dict):
        """Sends the appropriate command via socket to set the value of n variables, and waits for its 'ack' response.
        :param var_dict: Dictionary of variables to set (variable_name, value).
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        # construct unique command
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += '\n'  # new line is required for the command to finish
        # atomic commands send/rcv
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return self._is_ack(data)

    def _set_var(self, variable: str, value: Union[int, float]):
        """Sends the appropriate command via socket to set the value of a variable, and waits for its 'ack' response.
        :param variable: Variable to set.
        :param value: Value to set for the variable.
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str):
        """Sends the appropriate command to retrieve the value of a variable from the gripper, blocking until the
        response is received or the socket times out.
        :param variable: Name of the variable to retrieve.
        :return: Value of the variable as integer.
        """
        # atomic commands send/rcv
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)

        # expect data of the form 'VAR x', where VAR is an echo of the variable name, and X the value
        # note some special variables (like FLT) may send 2 bytes, instead of an integer. We assume integer here
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data} ({data.decode(self.ENCODING)}): does not match '{variable}'")
        value = int(value_str)
        return value

    @staticmethod
    def _is_ack(data: str):
        return data == b'ack'

    def _reset(self):
        """
        Reset the gripper.
        The following code is executed in the corresponding script function
        def rq_reset(gripper_socket="1"):
            rq_set_var("ACT", 0, gripper_socket)
            rq_set_var("ATR", 0, gripper_socket)

            while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                rq_set_var("ACT", 0, gripper_socket)
                rq_set_var("ATR", 0, gripper_socket)
                sync()
            end

            sleep(0.5)
        end
        """
        self._set_var(self.ACT, 0)
        # self._set_var(self.ATR, 0)
        while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
            self._set_var(self.ACT, 0)
            # self._set_var(self.ATR, 0)
        time.sleep(0.5)


    def activate(self, initialize: bool = True):
        self._set_var("MOD", 1)
        if not self.is_active():
            self._reset()
            while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
                time.sleep(0.01)

            self._set_var(self.ACT, 1)
            self._set_var(self.ATR, 0)
            
            
            time.sleep(1.0)
            while (not self._get_var(self.ACT) == 1 or not self._get_var(self.STA) == 3 or not self._get_var("MOD") == 1):
                print(self._get_var("MOD"))
                time.sleep(0.01)

        # auto-calibrate position range if desired
        if initialize:
            self.grip_off()

    def is_active(self):
        """Returns whether the gripper is active."""
        status = self._get_var(self.STA)
        return RobotiqHandE.GripperStatus(status) == RobotiqHandE.GripperStatus.ACTIVE

    def get_min_position(self) -> int:
        """Returns the minimum on the gripper can reach (open position)."""
        return self._min_position

    def get_max_position(self) -> int:
        """Returns the maximum position the gripper can reach (closed position)."""
        return self._max_position

    def get_open_position(self) -> int:
        """Returns what is considered the open position for gripper (minimum position value)."""
        return self.get_min_position()

    def get_closed_position(self) -> int:
        """Returns what is considered the closed position for gripper (maximum position value)."""
        return self.get_max_position()

    def is_open(self):
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self):
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()

    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        return self._get_var(self.POS)
    
    def detect_object(self):
        """Returns the current position as returned by the physical hardware."""
        return self._get_var(self.OBJ)
    
    def get_pre(self):
        return self._get_var(self.PRE)

    def grip(self, position, timeout=0, force=255):
        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))
        
        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_timeout, timeout, self._max_timeout)
        clip_for = clip_val(self._min_force, force, self._max_force)

        # moves to the given position with the given speed and force
        var_dict = OrderedDict([(self.POS, clip_pos), (self.SPE, clip_spe), (self.FOR, clip_for), (self.GTO, 1)])
       
        return self._set_vars(var_dict), clip_pos

    # def gripv2(self, position, timeout=0, force=255):
    #     def clip_val(min_val, val, max_val):
    #         return max(min_val, min(val, max_val))
        
    #     clip_pos = clip_val(self._min_position, position, self._max_position)
    #     clip_spe = clip_val(self._min_timeout, timeout, self._max_timeout)
    #     clip_for = clip_val(self._min_force, force, self._max_force)

    #     # moves to the given position with the given speed and force
    #     var_dict = OrderedDict([(self.POS, clip_pos), ("MOD", 1)])
    #     return self._set_vars(var_dict), clip_pos
    
    def grip_on(self, pres=0, timeout=100, force=100):
        return self.grip(pres, timeout, force)
    
    def grip_off(self, timeout=100, force=100):
        # print("ACT", self._get_var(self.ACT))
        # print("GTO", self._get_var(self.GTO))
        # # print("ATR", self._get_var(self.ATR))
        # # print("ADR", self._get_var(self.ADR))
        # print("FOR", self._get_var(self.FOR))
        # print("SPE", self._get_var(self.SPE))
        # print("POS", self._get_var(self.POS))
        return self.grip(255, timeout, force)

    def is_grip(self):
        """Returns whether the current position is considered as being fully closed."""
        press = self.get_current_position()
        # print(press)
        return press < 80.

if __name__ == "__main__":
    gripper = RobotiqHandE()
    gripper.connect("192.168.11.56", 63352)
    gripper.activate()

    s_time = time.time()
    e_time = time.time()-s_time
    gripper.grip(position=255,force=100)
    print("grip on")
    while e_time < 5:
        e_time = time.time()-s_time
        # print(e_time)

    gripper.grip(position=0, force=100)
    print("grip off")


