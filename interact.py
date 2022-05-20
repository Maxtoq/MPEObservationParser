import argparse
import json
import time
import numpy as np

from utils.make_env import make_env


class KeyboardActor:

    def __init__(self):
        pass

    def get_action(self):
        return [0.0, 0.0]


def run(args):
    # Load scenario config
    sce_conf = {}
    if args.sce_conf_path is not None:
        with open(args.sce_conf_path) as cf:
            sce_conf = json.load(cf)

    # Create environment
    env = make_env(
        args.env_path, 
        discrete_action=args.discrete_action, 
        sce_conf=sce_conf)

    actor = KeyboardActor()
    
    for ep_i in range(args.n_episodes):
        obs = env.reset()
        for step_i in range(args.episode_length):
            print("Step", step_i)
            print("Observations:", obs)
            # Get action
            action = actor.get_action()
            next_obs, rewards, dones, infos = env.step(action)
            print("Rewards:", rewards)

            time.sleep(args.step_time)
            env.render()

            if dones[0]:
                break
            obs = next_obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Scenario
    parser.add_argument("--env_path", default="env/coop_push_scenario_sparse.py",
                        help="Path to the environment")
    parser.add_argument("--sce_conf_path", default="configs/1a_1o_po_rel.json", type=str,
                        help="Path to the scenario config file")
    # Environment
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_true')
    # Render
    parser.add_argument("--step_time", default=0.1, type=float)

    args = parser.parse_args()
    run(args)