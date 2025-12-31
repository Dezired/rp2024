from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot
from transform import Affine
import pybullet as p

from scipy.spatial.transform import Rotation as R

import numpy as np
import os
import time
import yaml
from enum import Enum

class MoveDirection(Enum):
    BACKWARD = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    
class HandleEnvironment():
    def __init__(self, cfg, handleObjects, connection_mode=None):
        self.urdfPathRobot = os.path.join(cfg['ASSETS_PATH'], 'urdf', 'robot_without_gripper.urdf')
        self.urdfPathGoal = os.path.join(cfg['ASSETS_PATH'], 'objects', 'goals')
        self.hO = handleObjects
        self.IDs = {}
        if connection_mode is None:
            connection_mode = p.GUI if cfg.get("RENDER", False) else p.DIRECT
        self.bullet_client = BulletClient(connection_mode=connection_mode)
        self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if not cfg['RENDER']:
            self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.cfg = cfg
        self.robot = BulletRobot(bullet_client=self.bullet_client, urdf_path=self.urdfPathRobot)
        self.mDir = MoveDirection


    def reset(self):
        self.resetEnvironment()
        self.robotToStartPose()
        self.spawnGoals()
        self.spawnObjects()
        

    def resetEnvironment(self):
        self.bullet_client.resetSimulation()
        self.robot = BulletRobot(bullet_client=self.bullet_client, urdf_path=self.urdfPathRobot)
        self.robot.home()
        self.IDs = {}
        self.hO.reset() # reset objects and goals
        print("Environment resetted")

    def robotToStartPose(self):
        target_pose = Affine(translation=[0.5, 0.12, -0.1], rotation=[-np.pi, 0, np.pi/2]) # old start pose: [0.25, -0.34, -0.1]
        self.robot.lin(target_pose)

    def spawnGoals(self):
        goals = self.hO.generateGoals()
        if goals != None:
            for val, col in zip(goals.values(), self.cfg['COLOURS']):
                urdfPath = os.path.join(self.urdfPathGoal, f'goal_{col}.urdf')
                self.IDs[f'goal_{col}'] = []
                objID = self.bullet_client.loadURDF(
                                urdfPath,
                                val['pose'].translation,
                                val['pose'].quat,
                                flags=self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                            )
                self.IDs[f'goal_{col}'].append(objID)

    def spawnObjects(self):
        objects = self.hO.generateObjects()
        if objects is not None:
            for key, vals in objects.items():
                self.IDs[key] = []
                for pose in vals['poses']: # spawn all objects of same type and colour
                    objID = self.bullet_client.loadURDF(
                                    vals['urdfPath'],
                                    pose.translation,
                                    pose.quat,
                                    flags=self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                                )
                    self.IDs[key].append(objID)
            # Let objects settle
            for _ in range(100):
                self.bullet_client.stepSimulation()
                time.sleep(1/100)

    def getIDs(self):
        return self.IDs          

    def normalize(self, xVal, yVal, off=0.05):
        """Normalize a point to the range [-1, 1] with respect to the midpoint."""
        min_val_x, max_val_x = self.cfg['TABLE_CORDS']['x_min']-off, self.cfg['TABLE_CORDS']['x_max']+off
        min_val_y, max_val_y = self.cfg['TABLE_CORDS']['y_min']-off, self.cfg['TABLE_CORDS']['y_max']+off
        midpoint_x = (min_val_x + max_val_x) / 2
        midpoint_y = (min_val_y + max_val_y) / 2
        xN =  (xVal - midpoint_x) / (max_val_x - midpoint_x)
        yN =  (yVal - midpoint_y) / (max_val_y - midpoint_y)
        return xN, yN
    
    #def getStates(self):
    # # attention rework tableCords
    #    '''Returns normalized, flattened list as observation for robot, objects, and goals.'''
    #    objectStates, goalStates = [], []
    #    for key, ids in self.IDs.items():
    #        states = []
    #        for id in ids:
    #            pos, ori = self.bullet_client.getBasePositionAndOrientation(id)
    #            zAngle = R.from_quat(ori).as_euler('xyz')[2]
    #            # Normalize x and y positions
    #            norm_x = self.normalize(pos[0], self.hO.tableCords['x'][0], self.hO.tableCords['x'][1])
    #            norm_y = self.normalize(pos[1], self.hO.tableCords['y'][0], self.hO.tableCords['y'][1])
    #            states.extend([norm_x, norm_y, zAngle])
    #        if 'goal' in key:
    #            goalStates.extend(states)
    #        else:
    #            objectStates.extend(states)
    #    robotPose = self.robot.get_eef_pose().translation[:2]
    #    # Normalize robot pose
    #    norm_robot_x = self.normalize(robotPose[0], self.hO.tableCords['x'][0], self.hO.tableCords['x'][1])
    #    norm_robot_y = self.normalize(robotPose[1], self.hO.tableCords['y'][0], self.hO.tableCords['y'][1])
    #    paddedObjStates = np.pad(objectStates, (0, 3 * MAX_OBJECT_COUNT - len(objectStates)), constant_values=0)
    #    return np.concatenate([np.array([norm_robot_x, norm_robot_y]), paddedObjStates, np.array(goalStates)])

    def getStates(self):
        """Returns normalized, flattened list as observation for robot, objects, and goals."""
        self.positions = self.getPositions()
        goals = [] 
        objects = []
        for key, item in self.positions.items():
            if key == 'robot':
                robX, robY = item
                robXn, robYn = self.normalize(robX, robY)
            elif key.startswith('goal_'):
                for val in item.values():
                    goalX, goalY, goalZ = val
                    goalXn, goalYn = self.normalize(goalX, goalY)
                    goals.extend([goalXn, goalYn, goalZ])
            else: # objects
                for val in item.values():
                    objX, objY, objZ = val
                    objXn, objYn = self.normalize(objX, objY)
                    objects.extend([objXn, objYn, objZ])
        states = [robXn, robYn]
        states.extend(objects)
        states.extend(goals)
        return np.array(states)    
    
    #def getPositions(self):
    #    '''returns dict with nested list for dealing with position of robot, objects and goals individualy'''
    #    positionDict = {}
    #    for key, ids in self.IDs.items():
    #        positionDict[key] = {}
    #        for id in ids:
    #            pos, ori = self.bullet_client.getBasePositionAndOrientation(id)
    #            zAngle = R.from_quat(ori).as_euler('xyz')[2]
    #            positionDict[key][id] = [pos[0], pos[1], zAngle]
    #    positionDict['robot'] = self.robot.get_eef_pose().translation[:2]
    #    return positionDict
    
    # new function with padding:

    def getPositions(self):
        required_counts = {
            'goal_green': 1,
            'goal_red': 1,
            'plus_green': self.cfg['MAX_OBJECT_PER_TYPE'],
            'plus_red': self.cfg['MAX_OBJECT_PER_TYPE'],
            'cube_green': self.cfg['MAX_OBJECT_PER_TYPE'],
            'cube_red': self.cfg['MAX_OBJECT_PER_TYPE']
        }

        positionDict = {}
        for key, ids in self.IDs.items():
            count_needed = required_counts.get(key, 4)
            positionDict[key] = {}

            for i, obj_id in enumerate(ids):
                if i < count_needed:
                    pos, ori = self.bullet_client.getBasePositionAndOrientation(obj_id)
                    zAngle = R.from_quat(ori).as_euler('xyz')[2]
                    positionDict[key][obj_id] = [pos[0], pos[1], zAngle]

            existing_len = len(positionDict[key])
            for j in range(existing_len, count_needed):
                positionDict[key][f"dummy_{j}"] = [None, None, None]

        # Roboter-Position
        positionDict['robot'] = self.robot.get_eef_pose().translation[:2]
        return positionDict

    def performAction(self, action, step_size=0.01, debug=False):
        current_pose = self.robot.get_eef_pose()
        if debug:
            print(f"Current Pose: {current_pose.translation}, {current_pose.quat}")
        # Move 1 cm in XY
        dx, dy = {
            0: (-step_size,  0.0),
            1: ( step_size,  0.0),
            2: ( 0.0,   step_size),
            3: ( 0.0,  -step_size),
        }.get(action, (0.0, 0.0))
        if dx == dy == 0.0:
            return

        target_pose = Affine(
            translation=[current_pose.translation[0] + dx,
                        current_pose.translation[1] + dy,
                        self.cfg['TABLE_CORDS']['z']],   # keep z exactly the same 
            rotation= [-np.pi, 0, np.pi/2] # keep tcp straight to ground
        )
        self.robot.lin(target_pose) # or robot.ptp(target_pose)
    

    def robotLeavedWorkArea(self):
        '''returns True if robot out of Area''' # TODO
        [robotX, robotY] = self.robot.get_eef_pose().translation[:2]
        tableX = [self.cfg['TABLE_CORDS']['x_min'], self.cfg['TABLE_CORDS']['x_max']]
        tableY = [self.cfg['TABLE_CORDS']['y_min'], self.cfg['TABLE_CORDS']['y_max']]
        leaved = True
        if robotX < tableX[1] and robotX > tableX[0]-2.2: # check x
            if robotY < tableY[1] and robotY > tableY[0]: # check y
                leaved = False
        return leaved # DEBUG: False

    def objectOffTable(self):
        for key , values in self.IDs.items():
            if 'goal' not in key and 'robot' not in key:
                for id in values:
                    pos,_ = self.bullet_client.getBasePositionAndOrientation(id)
                    z = pos[2]
                    if z < 0: 
                        print(f"Error: Object {key} with ID {id} is off the table")
                        return True
        return False

    def checkMisbehaviour(self):
        '''check behaviour of robot and objects and return true if something misbehaves'''
        misbehaviour = self.objectOffTable() | self.robotLeavedWorkArea()
        if misbehaviour==True:
            print(f"Misbehaviour: {misbehaviour}")
        return misbehaviour


def drawStates(states):
    robotX, robotY, objX, objY, objZ, goalX, goalY, objZ = states
    # draw in with matplotlib
    import matplotlib.pyplot as plt
    plt.clf()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.plot(robotX, robotY, 'bo', label='Robot')
    plt.plot(objX, objY, 'ro', label='Object')
    plt.plot(goalX, goalY, 'go', label='Goal')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    from pprint import pprint
    from handleObjects import HandleObjects

    with open("src/config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    handleObjects = HandleObjects(config) 

    hEnv = HandleEnvironment(config, handleObjects)
    hEnv.spawnGoals()
    hEnv.spawnObjects()
    
    state_obj_z = hEnv.objectOffTable()
    print(f"State object z: {state_obj_z}")
    hEnv.robot.home()
    hEnv.robotToStartPose()
    ids = hEnv.getIDs()
    pprint(ids)
    input("Press Enter to continue...")
    states = hEnv.getStates()
    print('States:')
    pprint(states)
    drawStates(states)
    input("Press Enter to continue...")


