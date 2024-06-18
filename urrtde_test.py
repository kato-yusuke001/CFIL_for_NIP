from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# ロボット設定
robot_ip = "10.178.64.66"

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



dst = [0.650, -0.320, 0.200, 2.22, 2.22, -0.04]
rtde_c, rtde_r, gripper = connect(robot_ip)
# rtde_c = RTDEControlInterface(robot_ip)
# rtde_r = RTDEReceiveInterface(robot_ip)

current = rtde_r.getActualTCPPose()
print(type(current))
# dst = current
dst[0] = 0.650
dst[1] = -0.22

rtde_c.moveL(dst)
print("moveL to {} done".format(dst))

disconnect(rtde_c, rtde_r, gripper)
