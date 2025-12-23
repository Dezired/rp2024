from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot
from transform import Affine
import pybullet as p

import numpy as np
import os
import yaml

with open("src/config.yml", 'r') as stream:
    cfg = yaml.safe_load(stream)
urdfPathRobot = os.path.join(cfg['ASSETS_PATH'], 'urdf', 'robot_without_gripper.urdf')

def resetEnvironment(bulletClient, robot):
    bulletClient.resetSimulation()
    robot = BulletRobot(bullet_client=bulletClient, urdf_path=urdfPathRobot)
    robot.home()
    IDs = {}
    # hO.reset() # reset objects and goals
    print("Environment resetted")


def robotToStartPose(robot):
    target_pose = Affine(translation=[0.5, 0.12, -0.1], rotation=[-np.pi, 0, np.pi/2])
    robot.lin(target_pose)

def performAction(robot, action: int, step_size: float = 0.01, z_plane: float = -0.1):
    # 0: left (-x), 1: right (+x), 2: forward (+y), 3: back (-y)
    dx, dy = {
        0: (-step_size,  0.0),  # left
        1: ( step_size,  0.0),  # right
        2: ( 0.0,             step_size),  # forward
        3: ( 0.0,            -step_size),  # back
    }.get(action, (0.0, 0.0))

    if dx == dy == 0.0:
        return

    # get current pos
    current = robot.get_eef_pose()
    print(f"Current robot pose: {current.translation}, {current.quat}")
    # Keep current x,y, but project onto z_plane and freeze orientation
    cmd_x = float(current.translation[0])
    cmd_y = float(current.translation[1])
    fixed_quat = [-np.pi, 0, np.pi/2] # current.quat # TODO or [0, 0, 0, -1]
    # Update commanded (x, y) purely in the plane
    cmd_x += dx
    cmd_y += dy

    target_pose = Affine(
        translation=[cmd_x, cmd_y, z_plane],
        rotation=fixed_quat,
    )

    # For 1cm steps, ptp is usually enough and simple; lin also works
    # robot.lin(target_pose)
    robot.ptp(target_pose)

if __name__ == "__main__":
    bc = BulletClient(connection_mode=p.GUI)
    bc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    robot = BulletRobot(bullet_client=bc, urdf_path=urdfPathRobot)
    resetEnvironment(bc, robot)
    robotToStartPose(robot)
    while True:
        action = int(input("Enter action (0=left,1=right,2=forward,3=backward): "))
        performAction(robot, action)