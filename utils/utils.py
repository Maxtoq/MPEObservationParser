from utils.analyse import save_analyse
from utils.make_env import make_env
from utils.actors import RandomActor
import time
import json

# Save the sentences, actions and observations of the exercise as well as the analysis
def save(nb_agent, sentences, observations, actions):
    '''
    Save the data of the training (sentences, observations and actions) 
    in a json file. 
    Also calling the save_analysis() that will check if the user wants to save the analysis

    Input:
        nb_agents     :   number of agents in the environment
        sentences     :   all the sentences generated during the training
        observations  :   all observations generated during the training
        actions       :   every actions taken by the agents

    Output:
        Sentences_Generated.json
    '''
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
        for nb in range(nb_agent):
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

# Check the execution time of the program
def execution_time(args):
    '''
    Check the execution_time of the training depending on the number
    of episodes and the parser

    Print:  The total execution time
            The execution time per episode
            The execution time per step
    '''
    # Load scenario config
    sce_conf = {}
    if args.sce_conf_path is not None:
        with open(args.sce_conf_path) as cf:
            sce_conf = json.load(cf)

    # Create environment
    env, parser = make_env(
        args, 
        discrete_action=args.discrete_action, 
        sce_conf=sce_conf) 

    # Load initial positions if given
    if args.sce_init_pos is not None:
        with open(args.sce_init_pos, 'r') as f:
            init_pos_scenar = json.load(f)
    else:
        init_pos_scenar = None

    # The mouvement of the agents is randomised 
    actor = RandomActor(sce_conf["nb_agents"])

    # Save all the sentences generated
    sentences = [[],[]]
    # Save all the observations generated
    observations = []
    # Save all the actions genenrated
    action_list = []

    # Test timing execution speed
    t0 = time.time()
    
    for ep_i in range(args.n_episodes):
        # Reset the observation
        obs = env.reset(init_pos=init_pos_scenar)

        # Get the colors and the shapes of the episode
        colors = []
        shapes = []
        for object in env.world.objects :
                colors.append(object.num_color)
                shapes.append(object.num_shape)
                
        # For each step
        for step_i in range(args.episode_length):
            # Get action
            actions = actor.get_action()
            next_obs, rewards, dones, infos = env.step(actions)
            # Get sentence of agents
            for agent in range(sce_conf["nb_agents"]):
                #sentence = " "
                #sentence = observation.parse_obs(obs[agent],sce_conf)
                # Call the right parser
                if args.parser == "basic":
                    sentence = parser.parse_obs(obs[agent],sce_conf)
                if args.parser == 'strat':
                    sentence = parser.parse_obs(obs[agent],sce_conf, agent)
                sentences[agent].append(sentence)
            # Append the observation and the action of the step
            observations.append(obs)
            action_list.append(actions)

            obs = next_obs
    
    # Total time 
    t1 = time.time() - t0

    print("Execution time: " + str(t1))
    print("Execution time per episodes: " + str((t1)/args.n_episodes))
    print("Execution time per step: " + str((t1)/(args.n_episodes * args.episode_length)))


