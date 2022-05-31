import random
from math import sqrt
import argparse
import keyboard
import json
import time
import numpy as np

from utils.make_env import make_env
from env.coop_push_scenario_sparse import get_dist


class KeyboardActor:

    def __init__(self, n_agents):
        self.n_agents = n_agents

    def get_action(self):
        actions = np.zeros((self.n_agents, 2))
        actions[0] = np.array([0.0, 0.0])
        actions[1] = np.array([0.0, 0.0])

        if keyboard.is_pressed('z'):
            print('z')
            actions[0] += np.array([0.0, 0.5])
        if keyboard.is_pressed('s'):
            print('s')
            actions[0] += np.array([0.0, -0.5])
        if keyboard.is_pressed('q'):
            print('q')
            actions[0] += np.array([-0.5, 0])
        if keyboard.is_pressed('d'):
            print('d')
            actions[0] += np.array([0.5, 0.0])
        if keyboard.is_pressed('up arrow'):
            print('up')
            actions[1] += np.array([0.0, 0.5])
        if keyboard.is_pressed('down arrow'):
            print('down')
            actions[1] += np.array([0.0, -0.5])
        if keyboard.is_pressed('left arrow'):
            print('left')
            actions[1] += np.array([-0.5, 0])
        if keyboard.is_pressed('right arrow'):
            print('right')
            actions[1] += np.array([0.5, 0.0])

        return actions

class ObservationParserStrat:
    
    def __init__(self, args):
        self.args = args

    def parse_obs_strat(self, obs, sce_conf):
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []
        # If the action of pushing happens
        push = False

        # Position of the agent (at all time)
        sentence.append("Located")
        
        # North / South
        if obs[0][1] >= 0.32:
            sentence.append("North")
            position.append("North")
        if obs[0][1] < -0.32:
            sentence.append("South")
            position.append("South")
        
        # West / East
        if obs[0][0] >= 0.32:
            sentence.append("East")
            position.append("East")
        if obs[0][0] < -0.32:
            sentence.append("West")
            position.append("West")
        
        # Center
        elif len(sentence) == 1:
            sentence.append("Center")
            position.append("Center")
        

        # Position of the agent
        # For each agents 
        for agent in range(int(sce_conf['nb_agents'])-1):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents 
            place = place + agent*5 

            # If visible                                      
            if obs[0][place] == 1 :

                # Position
                sentence.append("You")
                collision = True
                 # North / South
                if obs[0][place+2] >= 0.15:
                    sentence.append("North")
                    collision = False
                if obs[0][place+2] < -0.15:
                    sentence.append("South")
                    collision = False
                
                # West / East
                if obs[0][place+1] >= 0.15:
                    sentence.append("East")
                    collision = False
                if obs[0][place+1] < -0.15:
                    sentence.append("West")
                    collision = False
                # If collision, don't print anything about the position
                if collision :
                    sentence.pop()

                # Is it moving ?
                if obs[0][place+4] > 0.5:
                    sentence.extend(["You","Search","North"])
                elif obs[0][place+4] < -0.5:
                    sentence.extend(["You","Search","South"])
                elif obs[0][place+3] > 0.5:
                    sentence.extend(["You","Search","East"])
                elif obs[0][place+3] < -0.5:
                    sentence.extend(["You","Search","West"])


        # Position of the objects
        # For each object 
        for object in range(int(sce_conf['nb_objects'])):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5 
            # 5 values for each other objects
            place = place + object*5 

            # If visible                                      
            if obs[0][place] == 1 :
                sentence.append("Object")
                 # North / South
                if obs[0][place+2] >= 0.25:
                    sentence.append("North")
                if obs[0][place+2] < -0.25:
                    sentence.append("South")
                
                # West / East
                if obs[0][place+1] >= 0.25:
                    sentence.append("East")
                if obs[0][place+1] < -0.25:
                    sentence.append("West")

                # Calculate the distance of the center of the object from the agent
                distance = obs[0][place+1]*obs[0][place+1] + obs[0][place+2]*obs[0][place+2]
                distance = sqrt(distance)
                
                """print("Distance: " + str(distance))
                print("Pos Obj: " + str(obs[0][place+1]) + " " + str(obs[0][place+2]))
                print("Vit Obj: " + str(obs[0][place+3]) + " " + str(obs[0][place+4]))"""

                # If collision
                if distance < 0.47:
                    sentence.extend(["I","Push","Object"])
                    push = True
                    # Calculate where the object was pushed 
                    # Based on its distance from the agent
                    if obs[0][place+2] > 0.20 and obs[0][place+2] < 0.50 :
                        sentence.append("North")
                    if obs[0][place+2] < -0.20 and obs[0][place+2] > -0.50 :
                        sentence.append("South")
                    if obs[0][place+1] > 0.20 and obs[0][place+1] < 0.50 :
                        sentence.append("East")
                    if obs[0][place+1] < -0.20 and obs[0][place+1] > -0.50:
                        sentence.append("West")
                
        
        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_objects'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5
            # 5 values for each objects 
            place = place + int(sce_conf['nb_objects'])*5 
            # 3 values for each landmark
            place = place + landmark*3

            # If visible
            if obs[0][place] == 1 :
                sentence.append("Landmark")
                
                # North / South
                if obs[0][place+2] >= 0.2:
                    sentence.append("North")
                if obs[0][place+2] < -0.2:
                    sentence.append("South")
                    
                # West / East
                if obs[0][place+1] >= 0.2:
                    sentence.append("East")
                if obs[0][place+1] < -0.2:
                    sentence.append("West")
                
                #If we are close to landmark
                elif (obs[0][place+2] < 0.2 and obs[0][place+2] >= -0.2 and
                    obs[0][place+1] < 0.2 and obs[0][place+2] >= -0.2):
                    # North / South
                    if obs[0][place+2] >= 0:
                        sentence.append("North")
                    if obs[0][place+2] < 0:
                        sentence.append("South")
                        
                    # West / East
                    if obs[0][place+1] >= 0:
                        sentence.append("East")
                    if obs[0][place+1] < 0:
                        sentence.append("West")

        # Search
        if not push:
            if obs[0][3] > 0.5:
                sentence.extend(["I","Search","North"])
            elif obs[0][3] < -0.5:
                sentence.extend(["I","Search","South"])
            elif obs[0][2] > 0.5:
                sentence.extend(["I","Search","East"])
            elif obs[0][2] < -0.5:
                sentence.extend(["I","Search","West"])

        print("Vitesse: " + str(obs[0][2]) + " " + str(obs[0][3]))

        return sentence


class ObservationParser:
    
    def __init__(self, args):
        self.args = args

    #Check the position of the agent to see if it is in a corner
    def check_position(self, obs):
        if (obs[0][0] >= 0.5 and (obs[0][1] >=0.5 or obs[0][1] <= -0.5) or
            obs[0][0] <= -0.5 and (obs[0][1] >=0.5 or obs[0][1] <= -0.5)):
            return True
        else:
            return False

    def parse_obs(self, obs, sce_conf):
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []

        #Generation of a NOT sentence ?
        not_sentence = 0
        if random.random() <= self.args.chance_not_sent:
            not_sentence = random.randint(1,3)
            print("REGARDE: " + str(not_sentence))


        # Position of the agent (at all time)
        sentence.append("Located")
        
        # North / South
        if obs[0][1] >= 0.32:
            sentence.append("North")
            position.append("North")
        if obs[0][1] < -0.32:
            sentence.append("South")
            position.append("South")
        
        # West / East
        if obs[0][0] >= 0.32:
            sentence.append("East")
            position.append("East")
        if obs[0][0] < -0.32:
            sentence.append("West")
            position.append("West")
        
        # Center
        elif len(sentence) == 1:
            sentence.append("Center")
            position.append("Center")
        

        # Position of the objects
        # For each object 
        for object in range(int(sce_conf['nb_objects'])):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5 
            # 5 values for each other objects
            place = place + object*5 

            # If not visible and not sentence
            if ((not_sentence == 1 or not_sentence == 3) 
                and obs[0][place] == 0):
                #if self.check_position(obs) :
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Object","Not"])
                    for word in position:
                        sentence.append(word)

            # If visible                                      
            if obs[0][place] == 1 :
                sentence.append("Object")
                 # North / South
                if obs[0][place+2] >= 0.25:
                    sentence.append("North")
                if obs[0][place+2] < -0.25:
                    sentence.append("South")
                
                # West / East
                if obs[0][place+1] >= 0.25:
                    sentence.append("East")
                if obs[0][place+1] < -0.25:
                    sentence.append("West")
        
        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_objects'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5
            # 5 values for each objects 
            place = place + int(sce_conf['nb_objects'])*5 
            # 3 values for each landmark
            place = place + landmark*3

            # If not visible and not sentence
            if ((not_sentence == 2 or not_sentence == 3) 
                and obs[0][place] == 0):
                #if self.check_position(obs):
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Landmark","Not"])
                    for word in position:
                        sentence.append(word)

            # If visible
            if obs[0][place] == 1 :
                sentence.append("Landmark")
                
                # North / South
                if obs[0][place+2] >= 0.2:
                    sentence.append("North")
                if obs[0][place+2] < -0.2:
                    sentence.append("South")
                    
                # West / East
                if obs[0][place+1] >= 0.2:
                    sentence.append("East")
                if obs[0][place+1] < -0.2:
                    sentence.append("West")
                
                #If we are close to landmark
                elif (obs[0][place+2] < 0.2 and obs[0][place+2] >= -0.2 and
                    obs[0][place+1] < 0.2 and obs[0][place+2] >= -0.2):
                    # North / South
                    if obs[0][place+2] >= 0:
                        sentence.append("North")
                    if obs[0][place+2] < 0:
                        sentence.append("South")
                        
                    # West / East
                    if obs[0][place+1] >= 0:
                        sentence.append("East")
                    if obs[0][place+1] < 0:
                        sentence.append("West")

        return sentence


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

    actor = KeyboardActor(sce_conf["nb_agents"])
    observation = ObservationParserStrat(args)
    
    for ep_i in range(args.n_episodes):
        obs = env.reset(init_pos=init_pos_scenar)
        for step_i in range(args.episode_length):
            print("Step", step_i)
            print("Observations:", obs)
            # Get action
            actions = actor.get_action()
            next_obs, rewards, dones, infos = env.step(actions)
            print("Rewards:", rewards)
            # Get sentence
            sentence = observation.parse_obs_strat(obs,sce_conf)
            print(sentence)

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
    parser.add_argument("--sce_conf_path", default="configs/1a_1o_po_rel.json", 
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

    args = parser.parse_args()
    run(args)