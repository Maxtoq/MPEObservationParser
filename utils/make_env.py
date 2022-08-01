import numpy as np
import json
import imp
import os
import re
from pathlib import Path
from shutil import copyfile

def make_env(args, sce_conf={}, discrete_action=False):
    scenario_path = args.env_path
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_path   :   path of the scenario script
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    from utils.render_multiagent import RenderMultiAgent

    # load scenario from script
    scenar_lib = imp.load_source('', scenario_path)
    scenario = scenar_lib.Scenario()

    # create world
    world = scenario.make_world(**sce_conf)
    # create multiagent environment
    env = RenderMultiAgent(world, scenario.reset_world, scenario.reward,
                        scenario.observation, 
                        done_callback=scenario.done if hasattr(scenario, "done")
                        else None, discrete_action=discrete_action)

    # Get the color and the shape
    colors = []
    shapes = []
    for object in env.world.objects :
            colors.append(object.num_color)
            shapes.append(object.num_shape)

    # Get parser
    if args.parser == "basic":
        parser = scenar_lib.ObservationParser(args, colors, shapes)
    if args.parser == 'strat':
        parser = scenar_lib.ObservationParserStrat(args, sce_conf, colors, shapes)

    return env, parser

def load_scenario_config(config, run_dir):
    sce_conf = {}
    if config.sce_conf_path is not None:
        copyfile(config.sce_conf_path, run_dir / 'sce_config.json')
        with open(config.sce_conf_path) as cf:
            sce_conf = json.load(cf)
            print('Special config for scenario:', config.env_path)
            print(sce_conf, '\n')
    return sce_conf