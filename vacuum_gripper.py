from typing import Any
import os, sys
import time


class VacuumGripper:
    def __init__(self, _io=None, _rtde_r=None) -> None:
        self.io = _io
        self.rtde_r = _rtde_r
        self.vacuum_valve_id = 0
        self.destruction_valve_id = 1
        self.pressure_input_id = 0

        self.io.setStandardDigitalOut(2,True)
    
    def disconnect(self):
        # TODO 吸着等を強制的にOFFにする？
        pass

    def reconnect(self):
        pass

    # ソレノイドバルブは24V通電で閉じる
    def ctrlValve(self, valve_id, val):
        self.io.setStandardDigitalOut(valve_id,val)

    #　供給バルブが開いていて、破壊バルブが閉じているときに真空発生
    def grip_on(self):
        self.ctrlValve(self.vacuum_valve_id, True)
        self.ctrlValve(self.destruction_valve_id, False)
    
    # 破壊バルブが開くと真空が破壊される.
    # ソレノイドバルブは長時間閉じていると焼き付いてしまうため、基本は開いておく
    def grip_off(self):
        self.ctrlValve(self.vacuum_valve_id, False)
        self.ctrlValve(self.destruction_valve_id, True)
        time.sleep(0.2)
        self.ctrlValve(self.destruction_valve_id, False)

    # 吸着圧の取得
    def getPressure(self):
        # self.io.setStandardDigitalOut(2,True)
        analog_input = self.rtde_r.getStandardAnalogInput0()
        # TODO:アナログ値（V）を吸着圧に変換? 
        # OFF時：約1.0V
        # ON時(未吸着):約1.24V（圧縮空気0.2MPa時）
        # ON時(吸着)：MAX:5V（圧縮空気の強さによる）    
        # self.io.setStandardDigitalOut(2,False)
        return analog_input

    # TODO:吸着の成否判定の実装
    def checkVacuum(self):
        pressure = self.getPressure()

        if(pressure > 80):
            return False
        else:
            return True
        

if __name__ == "__main__":
    from rtde_receive import RTDEReceiveInterface
    from rtde_io import RTDEIOInterface
    
    robot_ip = "192.168.11.100"
    rtde_r = RTDEReceiveInterface(robot_ip)
    io = RTDEIOInterface(robot_ip)
    vg = VacuumGripper(io, rtde_r)

    print("Before vacuum:", vg.getPressure())
    vg.grip_on()

    time.sleep(3)
    print("During vacuum:", vg.getPressure())

    # time.sleep(3)
    # print("During vacuum:", vg.getPressure())

    vg.grip_off()

    print("After vacuum:", vg.getPressure())

    time.sleep(3)
    print("Normal:", vg.getPressure())
