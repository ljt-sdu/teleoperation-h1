import casadi                                                                       
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin                
from pinocchio.robot_wrapper import RobotWrapper    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        # self.robot = pin.RobotWrapper.BuildFromURDF('../assets/h1_description/urdf/h1_with_hand.urdf', '../assets/h1_description/urdf')
        self.robot = pin.RobotWrapper.BuildFromURDF('../assets/h1_description/urdf/h1_with_hand.urdf', '../assets/h1_description/') # for test

        self.mixed_jointsToLockIDs = [  
                                        "right_hip_roll_joint",
                                        "right_hip_pitch_joint",
                                        "right_knee_joint",
                                        "left_hip_roll_joint",
                                        "left_hip_pitch_joint",
                                        "left_knee_joint",
                                        "torso_joint",
                                        "left_hip_yaw_joint",
                                        "right_hip_yaw_joint",

                                        "left_ankle_joint",
                                        "right_ankle_joint",

                                        "L_index_proximal_joint",
                                        "L_index_intermediate_joint",
                                        "L_middle_proximal_joint",
                                        "L_middle_intermediate_joint",
                                        "L_ring_proximal_joint",
                                        "L_ring_intermediate_joint",
                                        "L_pinky_proximal_joint",
                                        "L_pinky_intermediate_joint",
                                        "L_thumb_proximal_yaw_joint",
                                        "L_thumb_proximal_pitch_joint",
                                        "L_thumb_intermediate_joint",
                                        "L_thumb_distal_joint",
                                        
                                        "R_index_proximal_joint",
                                        "R_index_intermediate_joint",
                                        "R_middle_proximal_joint",
                                        "R_middle_intermediate_joint",
                                        "R_ring_proximal_joint",
                                        "R_ring_intermediate_joint",
                                        "R_pinky_proximal_joint",
                                        "R_pinky_intermediate_joint",
                                        "R_thumb_proximal_yaw_joint",
                                        "R_thumb_proximal_pitch_joint",
                                        "R_thumb_intermediate_joint",
                                        "R_thumb_distal_joint",

                                        "left_hand_joint",
                                        "right_hand_joint"    
                                      ]
   
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # for i, joint in enumerate(self.reduced_robot.model.joints):
        #     joint_name = self.reduced_robot.model.names[i]
        #     print(f"Joint {i}: {joint_name}, ID: {joint.id}")

        self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_elbow_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.2605 + 0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        
        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_elbow_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.2605 + 0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        

        self.init_data = np.zeros(self.reduced_robot.model.nq)

        # Initialize the Meshcat visualizer
        self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=True) 
        self.vis.loadViewerModel("pinocchio") 
        self.vis.displayFrames(True, frame_ids=[35, 75, 105, 106], axis_length = 0.15, axis_width = 5)
        self.vis.display(pin.neutral(self.reduced_robot.model))
        
        #for i in range(self.reduced_robot.model.nframes):
        #    frame = self.reduced_robot.model.frames[i]
        #    frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #    print(f"Frame ID: {frame_id}, Name: {frame.name}")
            
        # Enable the display of end effector target frames with short axis lengths and greater width.
        frame_viz_names = ['L_ee_target', 'R_ee_target']
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                      [0, 1, 0], [0.6, 1, 0],
                      [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 10
        for frame_viz_name in frame_viz_names:
            self.vis.viewer[frame_viz_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=axis_width,
                        vertexColors=True,
                    ),
                )
            )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.L_hand_id].inverse() * cpin.SE3(self.cTf_l)
                    ).vector[:3],
                    cpin.log6(
                        self.cdata.oMf[self.R_hand_id].inverse() * cpin.SE3(self.cTf_r)
                    ).vector[:3]
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.param_q_ik_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization = casadi.sumsqr(self.var_q)
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.param_q_ik_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(10 * self.totalcost + 0.001 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.001*self.regularization + 0.1*self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-4
            },
            'print_time':False
        }
        self.opti.solver("ipopt", opts)
        print(f"Initialize Arm_IK OK!")
        
    def adjust_pose(self, human_left_pose, human_right_pose, human_arm_length=0.55, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose
    

    #------ usr ljt add ------#
    def fk_fun(self, motorstate):
        """
        运动学正解算函数，计算给定关节角度下的左手和右手末端执行器在机器人中心坐标系中的位置和旋转。

        输入：
            q:机器人各关节的关节角度，是一个长度为 nq 的向量。包含左手和右手各4个关节。

        输出：
            left_trans_mat:左手末端在机器人中心坐标系中的变换矩阵(dim 4x4)
            right_trans_mat:右手末端在机器人中心坐标系中的变换矩阵(dim 4x4)
        """
        q = casadi.SX(motorstate)
        
        # 执行正向运动学计算
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        
        # 获取左手和右手末端执行器的帧ID
        left_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        right_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        
        # 提取左手和右手末端执行器的变换矩阵
        left_transform = self.cdata.oMf[left_hand_id]
        right_transform = self.cdata.oMf[right_hand_id]
        

        # 提取位置部分（从变换矩阵中取出平移部分，即前三个元素）
        left_position = left_transform.translation
        right_position = right_transform.translation
        
        # 提取旋转部分（从变换矩阵中取出旋转矩阵）
        left_rotation_matrix = left_transform.rotation
        right_rotation_matrix = right_transform.rotation
        
        print(f"left_position.type:{type(left_position)}\nleft_rotation_matrix.type:{type(left_rotation_matrix)}")
        print(f"left_position:{left_position}\nleft_rotation_matrix:{left_rotation_matrix}")
        
        # 使用numpy将旋转矩阵转换为四元数
        left_trans_mat = np.eye(4)
        left_trans_mat[:3, :3] = np.array(casadi.DM(left_rotation_matrix))
        left_trans_mat[:3, -1] = np.array(casadi.DM(left_position)).T
        
        right_trans_mat = np.eye(4)
        right_trans_mat[:3, :3] = np.array(casadi.DM(right_rotation_matrix))
        right_trans_mat[:3, -1] = np.array(casadi.DM(right_position)).T
 
        return left_trans_mat, right_trans_mat
    #------ usr ljt add ------#


    def ik_fun(self, left_pose, right_pose, motorstate=None, motorV=None):
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.vis.viewer['L_ee_target'].set_transform(left_pose)   # for visualization
        self.vis.viewer['R_ee_target'].set_transform(right_pose)  # for visualization

        # left_pose, right_pose = self.adjust_pose(left_pose, right_pose)

        self.opti.set_value(self.param_tf_l, left_pose)
        self.opti.set_value(self.param_tf_r, right_pose)

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            self.vis.display(sol_q)  # for visualization
            self.init_data = sol_q

            if motorV is not None:
                v =motorV * 0.0
            else:
                v = (sol_q-self.init_data ) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q,v,np.zeros(self.reduced_robot.model.nv))

            return sol_q, tau_ff ,True
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return sol_q, '',False


if __name__ == "__main__":
    arm_ik = Arm_IK()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.3, +0.2, 0.2]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.3, -0.2, 0.2]),
    )

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == 's':

        for i in range(100):
            L_tf_target.translation += np.array([0.002,  0.002, 0.002])
            R_tf_target.translation += np.array([0.002, -0.002, 0.002])

            sol_q ,tau_ff, flag = arm_ik.ik_fun(L_tf_target.homogeneous, R_tf_target.homogeneous)
            print(f"type(sol_q):{type(sol_q)}\t type(tau_ff):{type(tau_ff)}")
            time.sleep(0.02)
