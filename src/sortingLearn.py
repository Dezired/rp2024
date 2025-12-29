# from stable_baselines3 import PPO
# import os
#from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
#
#TIMESTEPS = 10000
#MODEL = "PPO_Hyperparameter_neu"
#
#modelsDir = f"/home/philipp/Documents/repos/rp2024/data/models/{MODEL}"
#if not os.path.exists(modelsDir):
#    os.makedirs(modelsDir)
#logDir = f"/home/philipp/Documents/repos/rp2024/data/logs/{MODEL}"
#if not os.path.exists(logDir):
#    os.makedirs(logDir)
#
#env = svpEnv()
#
##model = PPO('MlpPolicy', env, gamma = 0.99, ent_coef=0.01, verbose=1, tensorboard_log=logDir)
##model = PPO.load(f"/home/philipp/Documents/repos/rp2024/data/models/{MODEL}/100000.zip", env, gamma = 0.99, ent_coef=0.01, verbose=1, tensorboard_log=logDir) # use existing model
#model = PPO(
#    'MlpPolicy',
#    env,
#    gamma=0.95,  # Fokus auf kurzfristigere Belohnungen
#    ent_coef=0.02,  # Erhöhte Exploration
#    learning_rate=1e-4,  # Stabileres Lernen
#    n_steps=4096,  # Größere Batchgröße für stabilere Gradienten
#    clip_range=0.1,  # Engere Clipping-Range für stabilere Updates
#    max_grad_norm=1.0,  # Erhöhte Toleranz für Gradientenänderungen
#    batch_size=128,  # Größere Batchgröße für optimierte Lernzyklen
#    n_epochs=15,  # Mehr Trainingsepochen pro Update
#    verbose=1,
#    tensorboard_log=logDir
#)
#
#iters = 0
#while True:
#	iters += 1
#	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
#	model.save(f"{modelsDir}/{TIMESTEPS*iters}")

# python3 -m tensorboard.main --logdir=data/logs
     
from stable_baselines3 import PPO, A2C
from handleObjects import HandleObjects
from handleEnvironment import HandleEnvironment
from calcReward import CalcReward
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
import yaml
import os

with open("src/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

modelsDir = f"/home/philipp/Documents/repos/rp2024/data/models/{config['MODEL']}"
logDir = "/home/philipp/Documents/repos/rp2024/data/logs"
evalDir = "/home/philipp/Documents/repos/rp2024/data/evals"
os.makedirs(modelsDir, exist_ok=True)
os.makedirs(logDir, exist_ok=True)
os.makedirs(evalDir, exist_ok=True)

# Umgebung initialisieren
hO = HandleObjects(config)
hE = HandleEnvironment(config, hO)
calcR = CalcReward(config, hE)
env = svpEnv(config, hE, calcR)

model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=logDir)

iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=config['TIMESTEPS'], reset_num_timesteps=False, tb_log_name=config['MODEL'])
    model.save(f"{modelsDir}/{config['TIMESTEPS'] * iters}")

