import gymnasium as gym
import numpy as np
import yaml
from pprint import pprint

class sortingViaPushingEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	
	def __init__(self, config, hdlEnv, calcReward):
		super(sortingViaPushingEnv, self).__init__()
		self.action_space = gym.spaces.Discrete(4) # 4 directions (forward, backward, left, right)
		MAX_OBJECT_COUNT = config['MAX_OBJECT_PER_TYPE']*len(config['COLOURS'])*len(config['PARTS']) # max 4 of each object type and colour
		state_dim = config['ROBOT_STATE_COUNT'] + config['OBJECT_STATE_COUNT'] * MAX_OBJECT_COUNT + config['GOAL_STATE_COUNT'] * len(config['COLOURS']) # robot + max objects + goal states
		#state_dim = ROBOT_STATE_COUNT + OBJECT_STATE_COUNT * 1 + GOAL_STATE_COUNT * 1 # robot + max objects + goal states
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
											shape=(state_dim,), dtype=np.float64)
		self.cfg = config
		self.hdlEnv = hdlEnv
		self.calcReward = calcReward
		self.stepCount = 0
		self.startDistance = None
		self.score = 0

	def step(self, action):
		self.hdlEnv.performAction(action)
		self.terminated = self.calcReward.taskFinished()
		if self.hdlEnv.checkMisbehaviour():
			self.terminated = True
			self.reward = -1000
		else:
			self.reward = self.calcReward.calcReward()
		if self.stepCount >= self.cfg['MAX_STEPS']-1:
			self.truncated = True
		info = {'Step': self.stepCount, 'Reward': self.reward, 'Action': action, 'Terminated': self.terminated, 'Truncated': self.truncated}
		pprint(info)
		self.stepCount += 1
		observation = self.hdlEnv.getStates()
		#observation = self.calcReward.getStatePositions()
		

		return observation, self.reward, self.terminated, self.truncated, info
	
	def reset(self, seed=None):
		super().reset(seed=seed)
		self.stepCount = 0
		self.terminated = False
		self.truncated  = False
		self.prevReward = 0
		self.hdlEnv.resetEnvironment()
		self.hdlEnv.robotToStartPose()
		self.hdlEnv.spawnGoals()
		self.hdlEnv.spawnObjects()
		self.calcReward.reset()
		
        # create observation
		observation = self.hdlEnv.getStates() # robot state, object state, goal state (x,y|x,y,degZ|x,y,degZ)
		#observation = self.calcReward.getStatePositions()

		info = {}

		print("SortingViaPushingEnv resetted")
		return observation, info


if __name__ == "__main__":
	from handleEnvironment import HandleEnvironment
	from handleObjects import HandleObjects
	from calcReward import CalcReward

	with open("src/config.yml", 'r') as stream:
		config = yaml.safe_load(stream)

	handleObjects = HandleObjects(config)
	handleEnv = HandleEnvironment(config, handleObjects)
	handleEnv.spawnGoals()
	handleEnv.spawnObjects()
	calcReward = CalcReward(handleEnv)
	sortingEnv = sortingViaPushingEnv(config, handleEnv, calcReward)