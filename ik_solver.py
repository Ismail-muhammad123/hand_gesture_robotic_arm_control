import math
import numpy as np
from ikpy.chain import Chain
from ikpy.link import URDFLink
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import ikpy.utils.plot as plot_utils


class FiveDOF_IKSolver:
    def __init__(self):
        self.chain = Chain(name='5dof_arm', links=[
            URDFLink(name="base_rotation",  origin_translation=[0, 0, 0.05],  origin_orientation=[0, 0, 0], rotation=[0, 0, 1]),
            URDFLink(name="shoulder",       origin_translation=[0, 0, 0.05],  origin_orientation=[0,0, 0], rotation=[0, 1, 0]),
            URDFLink(name="elbow",          origin_translation=[0, 0, 0.12],  origin_orientation=[0, 0, 0], rotation=[0, 1, 0]),
            URDFLink(name="wrist_roll",     origin_translation=[0, 0, 0.09],  origin_orientation=[0, 0, 0], rotation=[0, 0, 1]),
            URDFLink(name="wrist_pitch",    origin_translation=[0, 0, 0.03],  origin_orientation=[0, 0, 0], rotation=[0, 1, 0]),
            URDFLink(name="gripper_fixed",  origin_translation=[0, 0, 0.05], origin_orientation=[0, 0, 0], rotation=None, joint_type="fixed"),
        ])
        # self.fig, self.ax = plot_utils.init_3d_figure()

    def solve_ik(self, x, y, z, orientation_euler=None):
        """
        Solve IK for target position (x, y, z) with optional orientation (roll, pitch, yaw) in radians.
        Returns joint angles in radians (excluding base fixed joint).
        """

        angles = self.chain.inverse_kinematics(
            target_position=[x,y,z],
            target_orientation = orientation_euler,
        )
        return angles
    
    def forward_kinematics(self, joint_angles):
        """
        Compute the forward kinematics from joint angles.
        Returns 4x4 end-effector transformation matrix.
        """
        angles_full = [0] + joint_angles
        return self.chain.forward_kinematics(angles_full)

# Example usageq
if __name__ == "__main__":
    solver = FiveDOF_IKSolver()
    target_pos = (0.05, 0.02, 0.2)  # in meters
    target_orientation = (np.pi/2.7, np.pi/4, 0)  # roll, pitch, yaw in radians (optional)

    joint_angles = solver.solve_ik(*target_pos, orientation_euler=target_orientation)
    print("Joint angles (degrees):", list(map(lambda r:round(math.degrees(r), 2),joint_angles.tolist()))[:5] )

    ee_pose = solver.forward_kinematics(joint_angles)
    print("Computed position: %s, original position : %s" % (ee_pose[:3, 3], target_pos))
    print("Computed position (readable) : %s" % [ '%.2f' % elem for elem in ee_pose[:3, 3] ])

   

