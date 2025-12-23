from stable_baselines3 import PPO, DQN, A2C
from handleObjects import HandleObjects
from handleEnvironment import HandleEnvironment
from calcReward import CalcReward
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
import numpy as np
import yaml

MODEL_PATH = '/home/philipp/Documents/repos/rp2024/data/models/A2C/80000'
with open("src/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

hO = HandleObjects(config)
hE = HandleEnvironment(config, hO)
calcR = CalcReward(config, hE)
ENV = svpEnv(config, hE, calcR)

def test_model(model_path, env):
    model = A2C.load(model_path)
    obs, _info = env.reset()
    
    for _ in range(1000):
        obs = np.array(obs)
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated:
            obs = env.reset()

if __name__ == "__main__":
    test_model(MODEL_PATH, ENV)