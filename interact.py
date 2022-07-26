import argparse
import keyboard
import json
import json
import time

from utils.embedding.ngram import embedding
from utils.parsers import ObservationParser, ObservationParserStrat, ObservationParserColor, ObservationParserStratColor, ObservationParserColorShape, ObservationParserStratColorShape
from utils.make_env import make_env
from utils.actors import KeyboardActor, RandomActor
from utils.analyse import analyze
from utils.render_option import Render_option
from utils.utils import save
from utils.utils import execution_time


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

    # Load initial positions if given
    if args.sce_init_pos is not None:
        with open(args.sce_init_pos, 'r') as f:
            init_pos_scenar = json.load(f)
    else:
        init_pos_scenar = None

    if args.actors == "manual" :
        actor = KeyboardActor(sce_conf["nb_agents"])
    elif args.actors == "random" :
        actor = RandomActor(sce_conf["nb_agents"])
    else:
        print("ERROR : Pick correct actors (random or manual)")
        exit(0)
    
    # Get the color and shape if needed
    colors = []
    shapes = []
    # Get the right parser based on the env
    observation = None
    match args.env_path:
        case "env/coop_push_scenario_sparse.py":
            if args.parser == "basic":
                observation = ObservationParser(args)
            elif args.parser == "strat":
                observation = ObservationParserStrat(args, sce_conf)
            else:
                print("ERROR : Pick correct parser (basic or strat)")
                exit(0)

        case "env/coop_push_scenario_sparse_color.py":
            # Get all the possible colors of the episode
            for object in env.world.objects :
                colors.append(object.num_color)

            if args.parser == "basic":
                observation = ObservationParserColor(args, colors)
            elif args.parser == "strat":
                observation = ObservationParserStratColor(args, sce_conf, colors)
            else:
                print("ERROR : Pick correct parser (basic or strat)")
                exit(0)

        case "env/coop_push_scenario_sparse_color_shape.py":
            # Get all the possible colors and shapes of the episode
            for object in env.world.objects :
                colors.append(object.num_color)
                shapes.append(object.num_shape)

            if args.parser == "basic":
                observation = ObservationParserColorShape(args, colors, shapes)
            elif args.parser == "strat":
                observation = ObservationParserStratColorShape(args, sce_conf)
            else:
                print("ERROR : Pick correct parser (basic or strat)")
                exit(0)

    if observation == None:
        print("ERROR : Pick correct env_path : env/coop_push_scenario_sparse + ")
        print(".py")
        print("_color.py")
        print("_color_shape.py")
        exit(0)


    # Save all the sentences generated
    sentences = [[],[]]
    # Save all the observations generated
    observations = []
    # Save all the actions genenrated
    action_list = []

    render_op = Render_option()

    for ep_i in range(args.n_episodes):
        obs = env.reset(init_pos=init_pos_scenar)
        for step_i in range(args.episode_length):
            print("Step", step_i)
            print("Observations:", obs)
            # Get action
            actions = actor.get_action()
            next_obs, rewards, dones, infos = env.step(actions)
            print("Rewards:", rewards)
            # Get sentence of the agents
            for agent in range(sce_conf["nb_agents"]):
                print(agent)
                if args.parser == "basic":
                    sentence = observation.parse_obs(obs[agent],sce_conf)
                if args.parser == 'strat':
                    sentence = observation.parse_obs(obs[agent],sce_conf, agent)
                print(sentence)
                sentences[agent].append(sentence)

            observations.append(obs)
            action_list.append(actions)

            # Get the render option
            range1, range2 = render_op.modify_option()

            time.sleep(args.step_time)
            env.render(range1,range2)

            if dones[0]:
                break
            obs = next_obs

                                                                        # ----------------- SOUS PROG ?? ----------------------- #

    # If we didn't already delete the Start and End tokens
    """if sentences[0][0][0] == '<SOS>':
        for i in range(2):
            for sentence in sentences[i]:
                # We delete first and last character
                sentence.pop(0)
                sentence.pop()"""
                    
    # Analysis of the sentences generated
    print("Would you like to see the analysis ?")
    print("Press A to see the analysis")
    print("Press any key to quit")
    if keyboard.read_key() == "a":
        analyze(sentences)
    else:
        # Clear the buffer
        keyboard.read_key()

    # Saves the data generated by the exercise in a json file
    print("Would you like to save the results ?")
    print("Press S to save")
    print("Press any key to quit")
    if keyboard.read_key() == "s":
        save(sce_conf,sentences,observations,action_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Scenario
    parser.add_argument("--env_path", default="env/coop_push_scenario_sparse_color_shape.py",
                        help="Path to the environment")
    parser.add_argument("--sce_conf_path", default="configs/2a_3o_po_rel.json", 
                        type=str, help="Path to the scenario config file")
    parser.add_argument("--sce_init_pos", default=None, 
                        type=str, help="Path to initial positions config file")
    # Environment
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_true')
    # Render
    parser.add_argument("--step_time", default=0.1, type=float)
    # Language
    parser.add_argument("--chance_not_sent", default=0.1, type=float)
    parser.add_argument("--parser", default="basic", type=str, help="Available parsers are 'basic' and 'strat'")
    # Action
    parser.add_argument("--actors", default="random", type=str, help="Available actors are 'random' or 'manual'")

    args = parser.parse_args()

    run(args)