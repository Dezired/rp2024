from stable_baselines3 import PPO, DQN
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
import numpy as np

MODEL_PATH = 'data/models/80000'
ENV = svpEnv()

def test_model(model_path, env):
    model = DQN.load(model_path)
    obs, _info = env.reset()
    
    for _ in range(1000):
        obs = np.array(obs)
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated:
            obs = env.reset()

if __name__ == "__main__":
    test_model(MODEL_PATH, ENV)