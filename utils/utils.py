from utils.analyse import save_analyse
from utils.make_env import make_env
from utils.actors import RandomActor
from utils.parsers import ObservationParser, ObservationParserStrat
import time
import json


def save(sce_conf, sentences, observations, actions):
    print("save pending")
    # Create a dictionnary out of the two variables
    dic = {}
    i = 0
    # For each step (each observation)
    for key in observations:
        # Generate the step
        dic['Step ' + str(i)] = {}
        # Add the state
        dic['Step ' + str(i)]['State'] = {}
        # For each agent
        for nb in range(sce_conf["nb_agents"]):
            agent_name = 'Agent_' + str(nb)
            # Add the observation of the agent
            dic['Step ' + str(i)][agent_name] = {}
            dic['Step ' + str(i)]['Agent_' + str(nb)]['Observation'] = {}
            dic['Step ' + str(i)]['Agent_' + str(nb)]['Observation'] = list(key[nb])
            dic['Step ' + str(i)]['Agent_' + str(nb)]['Sentence'] = sentences[nb][i]
            # Add action of the agent
            dic['Step ' + str(i)]['Agent_' + str(nb)]['Action'] = {}
            dic['Step ' + str(i)]['Agent_' + str(nb)]['Action'] = list(actions[i][nb])
        # Add one to the step counter  
        i += 1
    
    # Save the analysis of the agents
    analysis = save_analyse(sentences)
    if analysis != None:
        dic['Language analysis'] = analysis

    # Open file
    with open('Sentences_Generated.json', 'w', encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

    print("save success")

def execution_time(args):

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


    actor = RandomActor(sce_conf["nb_agents"])
    observation = ObservationParserStrat(args, sce_conf)

    # Save all the sentences generated
    sentences = [[],[]]
    # Save all the observations generated
    observations = []
    # Save all the actions genenrated
    action_list = []

    # Test timing execution speed
    t0 = time.time()
    
    for ep_i in range(args.n_episodes):
        obs = env.reset(init_pos=init_pos_scenar)
        for step_i in range(args.episode_length):
            # Get action
            actions = actor.get_action()
            next_obs, rewards, dones, infos = env.step(actions)
            # Get sentence of agents
            for agent in range(sce_conf["nb_agents"]):
                print(agent)
                sentence = observation.parse_obs(obs[agent],sce_conf,agent)
                print(sentence)
                sentences[agent].append(sentence)

            observations.append(obs)
            action_list.append(actions)

            obs = next_obs
    
    # Total time 
    t1 = time.time() - t0

    print("Execution time: " + str(t1))
    print("Execution time per episodes: " + str((t1)/args.n_episodes))
    print("Execution time per step: " + str((t1)/(args.n_episodes * args.episode_length)))


