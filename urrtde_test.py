from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# ロボット設定
robot_ip = "192.168.11.100"

# 通信開始
def connect(robot_ip):
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)
    gripper = None
    # gripper = RobotiqEpick()
    # gripper.connect(robot_ip, 63352)
    return rtde_c, rtde_r, gripper

# 通信終了
def disconnect(rtde_c, rtde_r, gripper):
    rtde_c.disconnect()
    rtde_r.disconnect()


rtde_c, rtde_r, gripper = connect(robot_ip)
# rtde_c = RTDEControlInterface(robot_ip)
# rtde_r = RTDEReceiveInterface(robot_ip)

current = rtde_r.getActualTCPPose()
print(type(current))
dst = current
dst[2] -= 0.01

rtde_c.moveL(dst)
print("moveL to {} done".format(dst))

disconnect(rtde_c, rtde_r, gripper)
