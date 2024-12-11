import numpy as np
from scipy.spatial.transform import Rotation as R

def transform_matrix_to_vector(transform):
    tvec = transform[:3, 3]
    rvec = R.from_matrix(transform[:3, :3]).as_rotvec()
    # 回転ベクトルの絶対値を180°以下にする。
    while True:
        dr = np.linalg.norm(rvec)
        if dr > np.radians(180):
            rvec = rvec * (dr - np.radians(360)) / dr
        else:
            break
    return np.hstack([tvec, rvec])

def rotvec_to_quat(rotvec):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html
    # (x, y, z, w) format
    return R.from_rotvec(rotvec).as_quat()

def rotvec_to_euler(rotvec, seq="xyz", degrees=False):
    # https://qiita.com/segur/items/1772f0b842bfabab3c6e
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html
    return R.from_rotvec(rotvec).as_euler(seq, degrees)

def pose_to_homogeneous(pose, epsilon=1e-6):
    B, T, D = pose.size()
    assert D == 6
    trans = pose[:, :, :3]
    rotvec = pose[:, :, 3:]
    output = torch.zeros((B, T, 4, 4), device=pose.device, dtype=pose.dtype)
    output[..., :3, :3] = rotvec_to_rotmat(rotvec, epsilon)
    output[..., :3, 3] = trans
    # Homogeneous line
    output[..., 3, :3] = 0.0
    output[..., 3, 3] = 1.0
    return output

def homogeneous_to_pose(H):
    trans = H[:, :, :3, 3]
    rotvec = rotmat_to_rotvec(H[:, :, :3, :3])
    pose = torch.cat([trans, rotvec], dim=-1)
    # print_tensor(trans, "trans")
    # print_tensor(rotvec, "rotvec")
    # print_tensor(pose, "pose")
    return pose

def rotvec_to_rotmat(rotvec, epsilon=1e-6):
    B, T, D = rotvec.size()
    assert D == 3
    rotvec = rotvec.view(B * T, D)
    theta = torch.norm(rotvec, dim=-1)
    small_theta = theta < epsilon
    # Rodrigues formula
    axis = rotvec / torch.clamp_min(theta[:, None], epsilon)
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    sin = torch.sin(theta)
    cos = torch.cos(theta)
    xs = kx * sin
    ys = ky * sin
    zs = kz * sin
    xyc = kx * ky * (1 - cos)
    xzc = kx * kz * (1 - cos)
    yzc = ky * kz * (1 - cos)
    xxc = kx * kx * (1 - cos)
    yyc = ky * ky * (1 - cos)
    zzc = kz * kz * (1 - cos)
    rotmat_rodrigues = torch.stack([
        1 - yyc - zzc, xyc - zs, xzc + ys,
        xyc + zs, 1 - xxc - zzc, yzc - xs,
        xzc - ys, yzc + xs, 1 - xxc - yyc],
        dim=-1).reshape(-1, 3, 3)
    # First order approximation for small theta
    xs, ys, zs = rotvec[:, 0], rotvec[:, 1], rotvec[:, 2]
    one = torch.ones_like(xs)
    rotmat_approx = torch.stack([
        one, -zs, ys,
        zs, one, -xs,
        -ys, xs, one],
        dim=-1).reshape(-1, 3, 3)
    # Rodrigues or approximation
    rotmat = torch.where(small_theta[:, None, None], rotmat_approx, rotmat_rodrigues)
    rotmat = rotmat.view(B, T, 3, 3)
    # print_tensor(rotmat, "rotmat")
    return rotmat

def rotmat_to_rotvec(R):
    unitquat = rotmat_to_unitquat(R)
    rotvec = unitquat_to_rotvec(unitquat)
    # print_tensor(R, "R", (0,0))
    # print_tensor(unitquat, "unitquat", (0,0))
    # print_tensor(rotvec, "rotvec", (0,0))
    return rotvec

def rotmat_to_unitquat(R):
    B, T, H, W = R.size()
    assert H == W == 3
    R = R.view(B * T, H, W)
    # 回転主軸を計算
    decision_matrix = torch.empty((B * T, 4), dtype=R.dtype, device=R.device)
    decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)  # 回転行列の対角成分
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)  # 回転行列の対角成分の和
    choices = decision_matrix.argmax(axis=1)  # 回転主軸
    # クォータニオンを計算
    quat = torch.empty((B * T, 4), dtype=R.dtype, device=R.device)
    # wが小さい場合
    ind_012 = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind_012]
    j = (i + 1) % 3
    k = (j + 1) % 3
    quat[ind_012, i] = 1 - decision_matrix[ind_012, -1] + 2 * R[ind_012, i, i]
    quat[ind_012, j] = R[ind_012, j, i] + R[ind_012, i, j]
    quat[ind_012, k] = R[ind_012, k, i] + R[ind_012, i, k]
    quat[ind_012, 3] = R[ind_012, k, j] - R[ind_012, j, k]
    # wが大きい場合
    ind_3 = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind_3, 0] = R[ind_3, 2, 1] - R[ind_3, 1, 2]
    quat[ind_3, 1] = R[ind_3, 0, 2] - R[ind_3, 2, 0]
    quat[ind_3, 2] = R[ind_3, 1, 0] - R[ind_3, 0, 1]
    quat[ind_3, 3] = 1 + decision_matrix[ind_3, -1]
    # 単位クォータニオンに変換
    unitquat = quat / torch.norm(quat, dim=1)[:, None]
    # print_tensor(unitquat[ind_012], "unitquat_012", (0,))
    # print_tensor(unitquat[ind_3], "unitquat_3", (0,))
    return unitquat.view(B, T, 4)

def unitquat_to_rotvec(quat, shortest_path=True):
    B, T, D = quat.size()
    assert D == 4
    quat = quat.view(B * T, D)
    quat = quat.clone()  # Need clone for differentiation
    # Make w > 0 to ensure 0 <= theta <= pi
    if shortest_path:
        quat[quat[:, 3] < 0] *= -1
    theta = torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3]) * 2
    small_theta = (torch.abs(theta) <= 1e-3)
    large_theta = ~small_theta
    scale = torch.empty(B * T, dtype=quat.dtype, device=quat.device)
    # scale = 2 + theta^2 / 12 + theta^4 * 7 / 2880
    scale[small_theta] = (2 + theta[small_theta] ** 2 / 12 + 7 * theta[small_theta] ** 4 / 2880)
    # scale = theta / sin_theta
    scale[large_theta] = (theta[large_theta] / torch.sin(theta[large_theta] / 2))
    rotvec = scale[:, None] * quat[:, :3]
    return rotvec.view(B, T, 3)