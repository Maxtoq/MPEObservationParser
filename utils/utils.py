

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
            dic['Step ' + str(i)]['Agent_' + str(nb)]['Action'] = str(actions[i][nb])
        # Add one to the step counter  
        i += 1
        

    # Open file
    with open('Sentences_Generated.json', 'w', encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

    print("save success")