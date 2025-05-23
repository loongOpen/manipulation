from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from teleop.constants_vuer import grd_yup2grd_zup, hand2inspire, tip_indices
from teleop.motion_utils import fast_mat_inv

float_array = np.load("/home/jyw/PycharmProjects/TeleVision/teleop/saved_data.npy")

head_data = np.array(float_array[:16])
left_data = np.array(float_array[16:32])
right_data = np.array(float_array[32:48])
left_landmarks=float_array[48:123].reshape(25,3)
right_landmarks=float_array[123:].reshape(25,3)

vuer_head_mat = head_data.reshape((4, 4))
vuer_left_wrist_mat = left_data.reshape((4, 4))
vuer_right_wrist_mat = right_data.reshape((4, 4))


head_mat = grd_yup2grd_zup @ vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
right_wrist_mat = grd_yup2grd_zup @ vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
left_wrist_mat = grd_yup2grd_zup @ vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

rel_left_wrist_mat = left_wrist_mat @ hand2inspire
rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]

rel_right_wrist_mat = right_wrist_mat @ hand2inspire  # wTr = wTh @ hTr
rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]

# homogeneous
left_fingers = np.concatenate([left_landmarks.copy().T, np.ones((1, left_landmarks.shape[0]))])
right_fingers = np.concatenate([right_landmarks.copy().T, np.ones((1, right_landmarks.shape[0]))])

# change of basis
left_fingers = grd_yup2grd_zup @ left_fingers
right_fingers = grd_yup2grd_zup @ right_fingers

rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
rel_left_fingers = (hand2inspire.T @ rel_left_fingers)[0:3, :].T
rel_right_fingers = (hand2inspire.T @ rel_right_fingers)[0:3, :].T

RetargetingConfig.set_default_urdf_dir('../assets')
with Path('inspire_hand.yml').open('r') as f:
    cfg = yaml.safe_load(f)
left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
left_retargeting = left_retargeting_config.build()
right_retargeting = right_retargeting_config.build()

left_qpos = left_retargeting.retarget(rel_left_fingers[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
right_qpos = right_retargeting.retarget(rel_right_fingers[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
print(np.rad2deg(left_qpos))
print(np.rad2deg(right_qpos))
# Generate some random 3D points
np.random.seed(0)
x = [*rel_left_fingers[:,0]]
y = [*rel_left_fingers[:,1]]
z = [*rel_left_fingers[:,2]]

# Create a new figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x, y, z, c='r', marker='o')

x = [*rel_right_fingers[:,0]]
y = [*rel_right_fingers[:,1]]
z = [*rel_right_fingers[:,2]]
ax.scatter(x, y, z, c='b', marker='o')

x = [*left_landmarks[:,0]]
y = [*left_landmarks[:,1]]
z = [*left_landmarks[:,2]]
ax.scatter(x, y, z, c='r', marker='*')

x = [*right_landmarks[:,0]]
y = [*right_landmarks[:,1]]
z = [*right_landmarks[:,2]]
ax.scatter(x, y, z, c='b', marker='*')


# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.axis('equal')

# Show the plot
plt.show()
