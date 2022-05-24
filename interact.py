import random
import argparse
from turtle import position
import keyboard
import json
import time
import numpy as np

from utils.make_env import make_env


class KeyboardActor:

    def __init__(self):
        pass

    def get_action(self):
        if keyboard.is_pressed('z'):
            print('z')
            return np.array([[0.0, 0.5]])
        elif keyboard.is_pressed('s'):
            print('s')
            return np.array([[0.0, -0.5]])
        elif keyboard.is_pressed('q'):
            print('q')
            return np.array([[-0.5, 0.0]])
        elif keyboard.is_pressed('d'):
            print('d')
            return np.array([[0.5, 0.0]])
        else:
            return np.array([[0.0, 0.0]])


class ObservationParser:

    
    def __init__(self):
        pass

    #Check the position of the agent to see if it is in a corner
    def check_position(self, obs):
        if (obs[0][0] >= 0.5 and (obs[0][1] >=0.5 or obs[0][1] <= -0.5) or
            obs[0][0] <= -0.5 and (obs[0][1] >=0.5 or obs[0][1] <= -0.5)):
            return True
        else:
            return False


    def parse_obs(self, obs, sce_conf):
        # Sentence generated
        sentence = ""
        # Absolute position of the agent
        position = ""

        #Generation of a NOT sentence ?
        not_sentence = False
        if random.randint(1,args.chance_not_sent) == 1:
            not_sentence = True

        # Position of the agent (at all time)
        sentence = "Located "
        # East
        if obs[0][0] >= 0.32:
           if obs[0][1] >= 0.32:
                position = "North East"
           elif obs[0][1] < -0.32:
                position = "South East"
           else:
               position = "East"

        # West
        elif obs[0][0] < -0.32:
           if obs[0][1] >= 0.32:
                position = "North West"
           elif obs[0][1] < -0.32:
                position = "South West"
           else:
              position = "West"

        # Center
        elif obs[0][0] > -0.32 and obs[0][0] <= 0.32:
               if obs[0][1] < -0.32:
                   position = "South"
               elif obs[0][1] >= 0.32:
                   position = "North"
               else:
                   position = "Center"
        
        sentence = sentence + position

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
            if not_sentence == True and obs[0][place] == 0:
                if self.check_position(obs) :
                    sentence = sentence + " Object Not " + position

            # If visible                                         
            if obs[0][place] == 1 :
                sentence = sentence + " Object "
                # East
                if obs[0][place+1] >= 0.30:
                    if obs[0][place+2] >= 0.30:
                            sentence = sentence + "North East"
                    elif obs[0][place+2] < -0.30:
                            sentence = sentence + "South East"
                    else:
                        sentence = sentence + "East"

                # West
                elif obs[0][place+1] < -0.30:
                    if obs[0][place+2] >= 0.30:
                            sentence = sentence + "North West"
                    elif obs[0][place+2] < -0.30:
                            sentence = sentence + "South West"
                    else:
                        sentence = sentence + "West"

                # North and South
                elif obs[0][place+1] > -0.30 and obs[0][place+1] <= 0.30:
                    if obs[0][place+2] < -0.30:
                        sentence = sentence + "South"
                    elif obs[0][place+2] >= 0.30:
                        sentence = sentence + "North"


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
            if not_sentence == True and obs[0][place] == 0:
                if self.check_position(obs):
                    sentence = sentence + " Landmark Not " + position

            # If visible
            if obs[0][place] == 1 :
                sentence = sentence + " Landmark "
                print(str(obs[0][place+1]) + " " + str(obs[0][place+2]))
                #East
                if obs[0][place+1] >= 0.32:
                    if obs[0][place+2] >= 0.32:
                            sentence = sentence + "North East"
                    elif obs[0][place+2] < -0.32:
                            sentence = sentence + "South East"
                    else:
                        sentence = sentence + "East"

                #West
                elif obs[0][place+1] < -0.32:
                    if obs[0][place+2] >= 0.32:
                            sentence = sentence + "North West"
                    elif obs[0][place+2] < -0.32:
                            sentence = sentence + "South West"
                    else:
                        sentence = sentence + "West"

                #North and South
                elif obs[0][place+1] > -0.32 and obs[0][place+1] <= 0.32:
                    if obs[0][place+2] < -0.32:
                        sentence = sentence + "South"
                    elif obs[0][place+2] >= 0.32:
                        sentence = sentence + "North"
                    else:
                        sentence = sentence + "Center" # If on top of the landmark                      OU NE RIEN METTRE ???

        # Tokenizing
        tokens = sentence.split(" ")

        return tokens


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
    observation = ObservationParser()
    
    for ep_i in range(args.n_episodes):
        obs = env.reset()
        for step_i in range(args.episode_length):
            print("Step", step_i)
            print("Observations:", obs)
            # Get action
            action = actor.get_action()
            next_obs, rewards, dones, infos = env.step(action)
            print("Rewards:", rewards)
            # Get sentence
            sentence = observation.parse_obs(obs,sce_conf)
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
    # Environment
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_true')
    # Render
    parser.add_argument("--step_time", default=0.1, type=float)
    # Language
    parser.add_argument("--chance_not_sent", default=10, type=int)

    args = parser.parse_args()
    run(args)