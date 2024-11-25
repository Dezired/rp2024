import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy

from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine


@hydra.main(version_base=None, config_path="config", config_name="tn_train_data")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    bullet_client = setup_bullet_client(cfg.render)

    env = instantiate(cfg.env, bullet_client=bullet_client)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    # the bounds for objects should be on the ground plane of the robots workspace
    t_bounds[2, 1] = t_bounds[2, 0]
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    oracle = instantiate(cfg.oracle)
    t_center = np.mean(t_bounds, axis=1)
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)

    for i in range(cfg.n_scenes):
        robot.home()
        robot.gripper.open()
        task = task_factory.create_task()
        task.setup(env, robot.robot_id)
        pose = oracle.solve(task)
        action = Affine.from_matrix(pose)
        pre_grasp_offset = Affine([0, 0, -0.1])
        pre_grasp_pose = action * pre_grasp_offset
        robot.ptp(pre_grasp_pose)
        robot.lin(action)
        robot.gripper.close()
        robot.lin(pre_grasp_pose)

        task.clean(env)

if __name__ == "__main__":
    main()
