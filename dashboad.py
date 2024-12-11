from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from dashboard_client import DashboardClient
import time

# robot_ip = "192.168.11.56"
robot_ip = "192.168.11.100"
# robot_ip = "192.168.11.30"
rtde_c = RTDEControlInterface(robot_ip)
rtde_r = RTDEReceiveInterface(robot_ip)
dashboad = DashboardClient(robot_ip)
dashboad.connect()

# Robot status: 0:電源OFF状態 1:アイドル状態orプログラム実行中 3:正常起動状態
print("Robot status", rtde_r.getRobotStatus())
print("safety status", rtde_r.getSafetyStatusBits())
# 1:正常状態？　7:非常停止ボタン押下 9:Fault 
print("safety mode", rtde_r.getSafetyMode()) 
# Safety status bits Bits 0-10: Is normal mode | Is reduced mode | Is protective stopped | Is recovery mode | Is safeguard stopped | Is system emergency stopped | Is robot emergency stopped | Is emergency stopped | Is violation | Is fault | Is stopped due to safety
# print(rtde_c.Flags)

# if(rtde_r.getSafetyMode() == 9):
#     dashboad.restartSafety()

# dashboad.closePopup()

# time.sleep(5)

# print(dashboad.isConnected())

# print(dashboad.powerOn())
# print("Robot status", rtde_r.getRobotStatus())
# print("safety status", rtde_r.getSafetyStatusBits())
# print("safety mode", rtde_r.getSafetyMode())
# time.sleep(5)
# print(dashboad.brakeRelease())

# # dashboad.unlockProtectiveStop()
# print(dashboad.isInRemoteControl())

# print("Robot status", rtde_r.getRobotStatus())
# print("safety status", rtde_r.getSafetyStatusBits())
# print("safety mode", rtde_r.getSafetyMode())

# time.sleep(20)
# print("Robot status", rtde_r.getRobotStatus())
# print("safety status", rtde_r.getSafetyStatusBits())
# print("safety mode", rtde_r.getSafetyMode())