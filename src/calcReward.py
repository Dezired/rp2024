import numpy as np

class CalcReward():
    def __init__(self, cfg, handleEnv):
        self.handleEnv = handleEnv
        self.distRobToGoal, self.distObjToGoal, self.distRobToObj  = None, None, None
        self.prevDistRobToGoal, self.prevDistObjToGoal, self.prevDistRobToObj = None, None, None
        self.nearObjectID, self.prevNearObjectID = None, None
        self.score = 0
        self.positions = self.handleEnv.getPositions()
        self.cfg = cfg
        self.firstStep = True

    def reset(self):
        self.distRobToGoal, self.distObjToGoal, self.distRobToObj  = None, None, None
        self.prevDistRobToGoal, self.prevDistObjToGoal, self.prevDistRobToObj = None, None, None
        self.nearObjectID, self.prevNearObjectID = None, None
        self.positions = self.handleEnv.getPositions()
        self.firstStep = True
        print("Reward calculator resetted")

    def calculateDistance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def checkObjectInsideGoal(self, objID):
        distDefInsideGoal = self.handleEnv.hO.goalWidths['x']/2-self.handleEnv.hO.objectWidth/2
        if self.getDistObjToGoal(objID) < distDefInsideGoal:
            return True
        else: # outside goal
            return False
        
    def getNearestObjToRob(self):
        while(True):
            if self.nearObjectID is None: # first or last step
                minDistance = float('inf')
                for key, positionsDict in self.positions.items():
                    if 'robot' not in key and 'goal' not in key:  # We don't want to compare robot to itself or to goal
                        # Check each position for an object (in case of multiple positions like 'plus_red')
                        for id, obj_position in positionsDict.items():
                            if obj_position != [None, None, None]:
                                distance = self.calculateDistance(self.positions['robot'], obj_position[:2])
                                if distance < minDistance: # new minDistance and objekt outside of goal
                                    if not self.checkObjectInsideGoal(id):
                                        minDistance = distance
                                        self.nearObjectID = id
                if self.nearObjectID is None:
                    return None, None        
                return minDistance, self.nearObjectID
            else:
                if self.checkObjectInsideGoal(self.nearObjectID): # check if nearest object is inside goal area
                        self.nearObjectID = None
                        dist = None
                else:  # Abstand berechnen
                    for key, positionsDict in self.positions.items():
                        if self.nearObjectID in positionsDict:
                            dist = self.calculateDistance(self.positions['robot'], positionsDict[self.nearObjectID][:2])
                            break
                    return dist, self.nearObjectID    

    def getDistRobotToObject(self): # allow switching of object
        minDistance = float('inf')
        self.nearObjectID = None 
        for key, positionsDict in self.positions.items():
            if 'robot' not in key and 'goal' not in key:
                for id, obj_position in positionsDict.items():
                    if type(id) == int:
                        distance = self.calculateDistance(self.positions['robot'], obj_position[:2])
                        if distance < minDistance: # new minDistance and objekt outside of goal
                            if not self.checkObjectInsideGoal(id):
                                minDistance = distance
                                self.nearObjectID = id
        if self.nearObjectID is None:
            return None, None 
        return minDistance, self.nearObjectID

    def getDistObjToGoal(self, objID):
        objName, objPos  = next(((obj, pos[objID]) for (obj, pos) in self.positions.items() if objID in self.positions[obj]), None)
        colour = objName.split('_')[1]
        _, goalPosDict = next(((obj, pos) for (obj, pos) in self.positions.items() if f'goal_{colour}' in obj), None)
        goalPos, = goalPosDict.values()
        return self.calculateDistance(objPos[:2], goalPos[:2])

    def getDistRobToGoal(self, objID):
        objName, _ = next(((obj, pos[objID]) for (obj, pos) in self.positions.items() if objID in self.positions[obj]), None)
        colour = objName.split('_')[1]
        _, goalPosDict = next(((obj, pos) for (obj, pos) in self.positions.items() if f'goal_{colour}' in obj), None)
        goalPos, = goalPosDict.values()
        return self.calculateDistance(self.positions['robot'], goalPos[:2])
    
    def getDistObjectsToGoal(self):
        distanceAllObjects = 0.0
        for objName, ids in self.handleEnv.IDs.items():
            if 'goal' not in objName:
                for id in ids:
                    distanceAllObjects += self.getDistObjToGoal(id)
        return distanceAllObjects

    def taskFinished(self):
        '''checks if all objects are inside their goal zones --> returns true otherwhise false'''
        for key, values in self.handleEnv.IDs.items():
            if 'goal' not in key and 'robot' not in key:
                for id in values:
                    if not self.checkObjectInsideGoal(id):
                        return False
        return True

    # def calcReward(self):
    #     self.positions = self.handleEnv.getPositions()
    #     self.prevNearObjectID = self.nearObjectID
    #     # dictance robot to nearest object 
    #     self.distRobToObj, self.nearObjectID = self.getNearestObjToRob()
    #     if self.handleEnv.objectOffTable():
    #         reward = -50
    #         return reward
    #     if self.nearObjectID is None:
    #         reward = 100
    #         return reward
    #     if (self.nearObjectID != self.prevNearObjectID):
    #         # set previous distance to new nearest obj
    #         self.prevDistRobToObj = self.distRobToObj
    #         self.prevDistObjToGoal = self.getDistObjToGoal(self.nearObjectID)
    #         self.prevDistRobToGoal = self.getDistRobToGoal(self.nearObjectID)
    #         reward = 15
    #         return reward
    #     # distance of that object to its goal
    #     self.distObjToGoal = self.getDistObjToGoal(self.nearObjectID)
    #     #distance of robot to goal for nearest object
    #     self.distRobToGoal = self.getDistRobToGoal(self.nearObjectID)
    #     #remeber distances for next step
    #     reward = -1
    #     if ((self.prevDistRobToObj - self.distRobToObj) > 0.0001) or (self.distRobToObj < 0.1):
    #         reward += 1.9
    #     #elif (self.distRobToObj - self.prevDistRobToObj) > 0.0001:
    #     #    reward -= 3.9
    #     if (self.prevDistObjToGoal - self.distObjToGoal) > 0.0001:
    #         reward += 10
    #     #elif (self.distObjToGoal - self.prevDistObjToGoal) > 0.0001:
    #     #    reward -= 10
    #     #if (self.distRobToGoal < self.distObjToGoal):
    #     #    if (self.prevDistRobToGoal - self.distRobToGoal) > 0.0001:
    #     #        reward -= 1
    #     #    elif (self.distRobToGoal - self.prevDistRobToGoal) > 0.0001:
    #     #        reward += 1

    #     print(f"Nearest Object:", next(((obj, pos[self.nearObjectID]) for (obj, pos) in self.positions.items() if self.nearObjectID in self.positions[obj]), None))
    
    #     self.prevDistRobToObj = self.distRobToObj 
    #     self.prevDistObjToGoal = self.distObjToGoal 
    #     self.prevDistRobToGoal = self.distRobToGoal 

    #     return reward

    def calcReward(self):
        self.positions = self.handleEnv.getPositions()
        self.prevNearObjectID = self.nearObjectID
        self.distRobToObj, self.nearObjectID = self.getDistRobotToObject() 
        self.distObjectsToGoal = self.getDistObjectsToGoal()
        reward = 0.0
        if self.firstStep:
            self.firstStep = False
            self.prevDistObjectsToGoals = self.distObjectsToGoal
            self.prevDistRobToObj = self.distRobToObj
        else:
            if self.distRobToObj is not None and self.prevDistRobToObj is not None:
                reward += (self.prevDistRobToObj - self.distRobToObj)
                print("reward rob to obj:", self.prevDistRobToObj - self.distRobToObj)
                reward += (self.prevDistObjectsToGoals - self.distObjectsToGoal)
                print("reward obj to goal:", self.prevDistObjectsToGoals - self.distObjectsToGoal)
                print("\n")

            if self.nearObjectID != self.prevNearObjectID and self.checkObjectInsideGoal(self.prevNearObjectID):
                reward += 50.0    

        # Update previous distances for next step
        self.prevDistRobToObj = self.distRobToObj
        self.prevDistObjectsToGoal = self.distObjectsToGoal
        return reward

    def calcReward2(self):
        step = 1 # 1 = move to obj, 2 = move obj to goal

        self.positions = self.handleEnv.getPositions()
        self.prevNearObjectID = self.nearObjectID
        # dictance robot to nearest object 
        self.distRobToObj, self.nearObjectID = self.getDistRobotToObject()
        if self.handleEnv.objectOffTable():
            reward = -50
            return reward
        if self.prevNearObjectID is None:
            self.startDistanceRobToObj = self.distRobToObj
            self.startDistanceObjectsToGoals = self.getDistObjectsToGoal()
        # if (self.nearObjectID != self.prevNearObjectID):
        #     # set previous distance to new nearest obj
        #     self.prevDistRobToObj = self.distRobToObj
        #     self.prevDistObjToGoal = self.getDistObjToGoal(self.nearObjectID)
        #     self.prevDistRobToGoal = self.getDistRobToGoal(self.nearObjectID)
        #     self.prevDistObjectsToGoals = self.getDistObjectsToGoal()
        #     if self.prevNearObjectID is not None:
        #         reward = 50
        #     else:
        #         self.startDistanceRobToObj = self.prevDistRobToObj
        #         self.startDistanceObjectsToGoals = self.prevDistObjectsToGoals
        #         reward = 0
        #     return reward
        
        print(f"Nearest Object:", next(((obj, pos[self.nearObjectID]) for (obj, pos) in self.positions.items() if self.nearObjectID in self.positions[obj]), None))
        # distance of that object to its goal
        self.distObjectsToGoal = self.getDistObjectsToGoal()
        #distance of robot to goal for nearest object
        #self.distRobToGoal = self.distRobToObj # self.getDistRobToGoal(self.nearObjectID)
        if self.distRobToObj == float('inf'):
            reward = 100
            return reward
        reward = 0
        if step == 1:
            reward += self.startDistanceRobToObj - self.distRobToObj
            reward += self.startDistanceObjectsToGoals - self.distObjectsToGoal


        self.prevDistRobToObj = self.distRobToObj
        self.prevDistObjToGoal = self.distObjToGoal
        self.prevDistRobToGoal = self.distRobToGoal

        return reward
    
    def getStatePositions(self):
        '''Returns normalized, flattened list as observation for robot, nearest object, and its corresponding goal'''
        robotState = self.positions['robot']
        nearestObjectState = [0.0, 0.0, 0.0]
        nearestGoalState = [0.0, 0.0, 0.0]
        key1 = None

        # Normalize robot position
        norm_robot_x = self.handleEnv.normalize(robotState[0], self.cfg['TABLE_CORDS']['x_min'], self.cfg['TABLE_CORDS']['x_max'])
        norm_robot_y = self.handleEnv.normalize(robotState[1], self.cfg['TABLE_CORDS']['y_min'], self.cfg['TABLE_CORDS']['y_max'])
        robotState = [norm_robot_x, norm_robot_y]

        # Find nearest object and normalize its position
        for key, positionsDict in self.positions.items():
            if self.nearObjectID in positionsDict:
                nearestObjectState = positionsDict[self.nearObjectID]
                key1 = key

        if key1:
            nearestObjectState[0] = self.handleEnv.normalize(nearestObjectState[0], self.cfg['TABLE_CORDS']['x_min'], self.cfg['TABLE_CORDS']['x_max'])
            nearestObjectState[1] = self.handleEnv.normalize(nearestObjectState[1], self.cfg['TABLE_CORDS']['y_min'], self.cfg['TABLE_CORDS']['y_max'])
        
            # Find the goal corresponding to the object's color and normalize its position
            colour = key1.split('_')[1]
            _, goalPosDict = next(((obj, pos) for (obj, pos) in self.positions.items() if f'goal_{colour}' in obj), None)
            if goalPosDict:
                goalPos, = goalPosDict.values()
                nearestGoalState[0] = self.handleEnv.normalize(goalPos[0], self.cfg['TABLE_CORDS']['x_min'], self.cfg['TABLE_CORDS']['x_max'])
                nearestGoalState[1] = self.handleEnv.normalize(goalPos[1], self.cfg['TABLE_CORDS']['y_min'], self.cfg['TABLE_CORDS']['y_max'])

        return np.concatenate([robotState, nearestObjectState, nearestGoalState])

    
    # def calcReward2(self): # use euclidian distance and reward pushing object into goal, punish switching objects
    #     reward = 0
    #     self.positions = self.handleEnv.getPositions()
    #     self.prevNearObjectID = self.nearObjectID
    #     self.distRobToObj, self.nearObjectID = self.getNearestObjToRob()
    #     self.distObjToGoal = self.getDistObjToGoal(self.nearObjectID)
    #     self.distRobToGoal = self.getDistRobToGoal(self.nearObjectID)
    #     if (self.nearObjectID != self.prevNearObjectID): # new object --> reset treshhold so euclidian reward starts with 0
    #         self.prevDistRobToObj = self.distRobToObj
    #         self.prevDistObjToGoal = self.distObjToGoal
    #         self.prevDistRobToGoal = self.distRobToGoal
    #         reward =+ 15 # award one more object in goal

    #     rewardRobToObj = self.prevDistRobToObj - self.distRobToObj
    #     rewardObjToGoal = self.prevDistObjToGoal - self.distObjToGoal
    #     rewardRobToGoal = self.prevDistRobToGoal - self.distRobToGoal
    #     print(f"Nearest Object:", next(((obj, pos[self.nearObjectID]) for (obj, pos) in self.positions.items() if self.nearObjectID in self.positions[obj]), None))
    #     return reward + (3*rewardRobToObj + 2*rewardObjToGoal + rewardRobToGoal) # base reward + euclidian rewards

if __name__ == "__main__":
    from handleEnvironment import HandleEnvironment
    from handleObjects import HandleObjects
    import yaml
    with open("src/config.yml", 'r') as stream:
        config = yaml.safe_load(stream)

    hO = HandleObjects(config)

    hEnv = HandleEnvironment(config, hO)
    hEnv.spawnGoals()
    hEnv.spawnObjects()

    calcRew = CalcReward(config, hEnv)

    reward = calcRew.calcReward()
    print(f"Calculated reward: {reward}")
    input("Press Enter to continue...")
    # state = calcRew.getStatePositions()
    # print(f"State positions: {state}")  