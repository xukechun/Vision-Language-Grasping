import time
import glob
import pybullet as pb
import pybullet_data
import numpy as np
from operator import itemgetter
from scipy.spatial.transform import Rotation as R
import cameras
from constants import PIXEL_SIZE, WORKSPACE_LIMITS, LANG_TEMPLATES, LABEL, GENERAL_LABEL, COLOR_SHAPE, FUNCTION, LABEL_DIR_MAP, KEYWORD_DIR_MAP

class Environment:
    def __init__(self, gui=True, time_step=1 / 240):
        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """

        self.time_step = time_step
        self.gui = gui
        self.pixel_size = PIXEL_SIZE
        self.obj_ids = {"fixed": [], "rigid": []}
        self.agent_cams = cameras.RealSenseD435.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.bounds = WORKSPACE_LIMITS
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        # Start PyBullet.
        self._client_id = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(time_step)

        if gui:
            target = pb.getDebugVisualizerCamera()[11]
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target,
            )

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(pb.getBaseVelocity(i, physicsClientId=self._client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = pb.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self._client_id
                )
                dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    def obj_info(self, obj_id):
        """Environment info variable with object poses, dimensions, and colors."""

        pos, rot = pb.getBasePositionAndOrientation(
            obj_id, physicsClientId=self._client_id
        )
        dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
        info = (pos, rot, dim)
        return info    

    def generate_lang_goal(self):
        prob = np.array([0.4, 0.2, 0.1, 0.1, 0.2])
        template_id = np.random.choice(a=range(len(LANG_TEMPLATES)), size=1, p=prob)[0]

        if template_id == 0:
            id = np.random.choice(range(len(LABEL)), 1)[0]
            keyword = LABEL[id]
            self.target_obj_dir = [LABEL_DIR_MAP[id]]
            self.target_obj_lst = self.target_obj_dir
        else:
            if template_id == 1:
                id = np.random.choice(range(len(GENERAL_LABEL)), 1)[0]
                keyword = GENERAL_LABEL[id]
                self.target_obj_dir = KEYWORD_DIR_MAP[keyword]
            elif template_id == 2:
                id = np.random.choice(range(len(COLOR_SHAPE)), 1)[0]
                keyword = COLOR_SHAPE[id]
                self.target_obj_dir = KEYWORD_DIR_MAP[keyword]
            elif template_id == 3:
                id = np.random.choice(range(len(COLOR_SHAPE)), 1)[0]
                keyword = COLOR_SHAPE[id]
                self.target_obj_dir = KEYWORD_DIR_MAP[keyword]
            elif template_id == 4:
                id = np.random.choice(range(len(FUNCTION)), 1)[0]
                keyword = FUNCTION[id]
                self.target_obj_dir = KEYWORD_DIR_MAP[keyword]
            
            if len(self.target_obj_dir) > 3:
                batch = np.random.choice(range(len(self.target_obj_dir)), 2, replace=False) 
                self.target_obj_lst = list(itemgetter(*batch)(self.target_obj_dir))
            else:
                self.target_obj_lst = self.target_obj_dir

        self.lang_goal = LANG_TEMPLATES[template_id].format(keyword=keyword)
        pb.addUserDebugText(text=self.lang_goal, textPosition=[0.8, -0.2, 0], textColorRGB=[0, 0, 1], textSize=2)
        
        return self.lang_goal

    def get_target_id(self):
        return self.target_obj_ids

    def add_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].append(obj_id)

    def remove_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].remove(obj_id)

    def save_objects(self):
        """Save states of all rigid objects. If this is unstable, could use saveBullet."""
        success = False
        while not success:
            success = self.wait_static()
        object_states = []
        for obj in self.obj_ids["rigid"]:
            pos, orn = pb.getBasePositionAndOrientation(obj)
            linVel, angVel = pb.getBaseVelocity(obj)
            object_states.append((pos, orn, linVel, angVel))
        return object_states

    def restore_objects(self, object_states):
        """Restore states of all rigid objects. If this is unstable, could use restoreState along with saveBullet."""
        for idx, obj in enumerate(self.obj_ids["rigid"]):
            pos, orn, linVel, angVel = object_states[idx]
            pb.resetBasePositionAndOrientation(obj, pos, orn)
            pb.resetBaseVelocity(obj, linVel, angVel)
        success = self.wait_static()
        return success

    def wait_static(self, timeout=3):
        """Step simulator asynchronously until objects settle."""
        pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            pb.stepSimulation()
        print(f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False

    def reset(self):
        self.obj_ids = {"fixed": [], "rigid": []}
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        if self.gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # Load workspace
        self.plane = pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        self.workspace = pb.loadURDF(
            "assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True,
        )
        pb.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        pb.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # Load UR5e
        self.ur5e = pb.loadURDF(
            "assets/ur5e/ur5e.urdf", basePosition=(0, 0, 0), useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(pb.getNumJoints(self.ur5e)):
            info = pb.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        pb.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)

        self.setup_gripper()

        # Move robot to home joint configuration.
        success = self.go_home()
        self.close_gripper()
        self.open_gripper()

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering.
        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )

    def setup_gripper(self):
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = pb.loadURDF(
            "assets/ur5e/gripper/robotiq_2f_85.urdf",
            ee_position,
            pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.ee_tip_z_offset = 0.1625
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(pb.getNumJoints(self.ee)):
            info = pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                pb.changeDynamics(
                    self.ee, joint_id, lateralFriction=0.9
                )
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )
        self.ee_constraint = pb.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(self.ee_constraint, maxForce=10000)
        pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: right
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)

    def step(self, pose=None):
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        done = False
        if pose is not None:
            success, grasped_obj_id, pos_dist = self.grasp(pose)
            # Grasping fails
            if not success:
                reward = -1
            else:
                if grasped_obj_id in self.target_obj_ids:
                    reward = 2
                    done = True
                else:
                    max_pos_dist = np.sqrt((WORKSPACE_LIMITS[0][1]-WORKSPACE_LIMITS[0][0]) ** 2 + (WORKSPACE_LIMITS[1][1]-WORKSPACE_LIMITS[1][0]) ** 2)
                    reward = - pos_dist / max_pos_dist

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            pb.stepSimulation()

        return reward, done

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def __del__(self):
        pb.disconnect()

    def get_link_pose(self, body, link):
        result = pb.getLinkState(body, link)
        return result[4], result[5]

    def add_objects(self, num_obj, workspace_limits):
        """Randomly dropped objects to the workspace"""
        mesh_list = glob.glob("assets/simplified_objects/*.urdf")

        # get target object
        target_mesh_list = []
        for target_obj in self.target_obj_lst:
            target_mesh_file = "assets/simplified_objects/" + target_obj + ".urdf"
            target_mesh_list.append(target_mesh_file)
        for obj in self.target_obj_dir:
            obj_mesh_file = "assets/simplified_objects/" + obj + ".urdf"
            mesh_list.remove(obj_mesh_file)

        obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj-len(self.target_obj_lst))

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        self.target_obj_ids = []

        with open("cases/" + "temp.txt", "w") as out_file:
            out_file.write("%s\n" % self.lang_goal)
            # add target objects
            for target_mesh_file in target_mesh_list:
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
                    + 0.1
                )
                object_position = [drop_x, drop_y, 0.2]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                body_id = pb.loadURDF(
                    target_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                # pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
                body_ids.append(body_id)
                self.target_obj_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        target_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )

            # add other objects
            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
                    + 0.1
                )
                object_position = [drop_x, drop_y, 0.2]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        curr_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )


        return body_ids, True

    def add_object_push_from_file(self, file_name, switch=None):
        success = True
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            self.lang_goal = file_content[0].split('\n')[0]
            target_obj = file_content[1].split()
            self.target_obj_ids = [4 + int(i) for i in target_obj]
            num_obj = len(file_content) - 2
            obj_files = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx+2].split()
                obj_file = file_content_curr_object[0]
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            body_id = pb.loadURDF(
                curr_mesh_file,
                object_position,
                pb.getQuaternionFromEuler(object_orientation),
                flags=pb.URDF_ENABLE_SLEEPING
            )
            self.add_object_id(body_id)
            success &= self.wait_static()
            success &= self.wait_static()

        # give time to stop
        for _ in range(5):
            pb.stepSimulation()

        return success, self.lang_goal

    def get_true_object_pose(self, obj_id):
        pos, ort = pb.getBasePositionAndOrientation(obj_id)
        position = np.array(pos).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(ort)
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        return transform   

    def get_true_object_poses(self):
        transforms = dict()
        for obj_id in self.obj_ids["rigid"]:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.01, timeout=3):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(
                [
                    pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                    for i in self.ur5e_joints
                ]
            )
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 0.05):
                # give time to stop
                for _ in range(5):
                    pb.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            pb.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = pb.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, is_push=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_force:
                force = np.sum(
                    np.abs(np.array(pb.getJointState(self.ur5e, self.ur5e_ee_id)[2]))
                )
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False    
        if is_push:
            speed /= 5
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def grasp(self, pose, speed=0.005):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:]
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform

        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        # Execute 6-dof grasping.
        grasped_obj_id = None
        min_pos_dist = None
        self.open_gripper()
        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over, rot))
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot, speed)
            success &= self.is_gripper_closed
            
            if success: # get grasp object id
                max_height = -0.0001
                for i in self.obj_ids["rigid"]:
                    height = self.info[i][0][2]
                    if height >= max_height:
                        grasped_obj_id = i
                        max_height = height
                pos_dists = []
                for target_obj_id in self.target_obj_ids:
                    pos_dist = np.linalg.norm(np.array(self.info[grasped_obj_id][0]) - np.array(self.info[target_obj_id][0]))
                    pos_dists.append(pos_dist)
                min_pos_dist = min(pos_dists)

        if success:
            success = self.move_joints(self.drop_joints1)
            # success &= self.is_gripper_closed
            self.open_gripper(is_slow=True)
        self.go_home()

        print(f"Grasp at {pose}, the grasp {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success, grasped_obj_id, min_pos_dist

    def open_gripper(self, is_slow=False):
        self._move_gripper(self.gripper_angle_open, is_slow=is_slow)

    def close_gripper(self, is_slow=True):
        self._move_gripper(self.gripper_angle_close, is_slow=is_slow)

    @property
    def is_gripper_closed(self):
        gripper_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]
        return gripper_angle < self.gripper_angle_close_threshold

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        t0 = time.time()
        prev_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]

        if is_slow:
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            for _ in range(10):
                pb.stepSimulation()
            while (time.time() - t0) < timeout:
                current_angle = pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    pb.stepSimulation()
        # maintain the angles
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        for _ in range(10):
            pb.stepSimulation()
