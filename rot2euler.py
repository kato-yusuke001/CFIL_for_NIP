from scipy.spatial.transform import Rotation
import pandas as pd
import argparse
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def rot2euler(rot):
    R = Rotation.from_rotvec(rot)
    return R.as_euler('xyz', degrees=True)

def euler2rot(euler):
    R = Rotation.from_euler('xyz', euler, degrees=True)
    return R.as_rotvec()

def read_csv(file_path):
    # with StringIO(file_path) as csvfile:
    df = pd.read_csv(file_path, header=0, dtype=float)
    return df


 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", type=str, default="test.csv")
    args = parser.parse_args()

    file_path = args.file_path
    result = read_csv(file_path)
    bt_rotVecs = result.loc[:, 'bottleneck_pose_rx':'bottleneck_pose_rz'].values.tolist()
    before_rotVecs = result.loc[:, 'before_pose_rx':'before_pose_rz'].values.tolist()
    after_rotVecs = result.loc[:, 'after_pose_rx':'after_pose_rz'].values.tolist()

    for bt, be, af in zip(bt_rotVecs, before_rotVecs, after_rotVecs):
        print(bt, be, af)
        print(rot2euler(bt), rot2euler(be), rot2euler(af))
        print("=====================================")  