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

"""class TrackingAgent:

    directions = []
    time = 0

    def __init__(self):
        pass

    def update_direction(self, direction):
        if self.directions == direction:
            time = time + 1
        else:
            self.directions = direction
        
        if time == 3:
            return True
        else:
            return False"""


class ObservationParserStrat:
    
    def __init__(self, args):
        self.args = args
        self.directions = []
        self.time = 0

        # Initialization of the world map
        # To track the agent
        self.world = []
        for line in range(8):
            newLine = []
            for col in range(8):
                newLine.append(0)
            self.world.append(newLine)

        # Initialization of the area map
        # Each 0 is an area (North, South, West, East)
        # That has not been fully discovered
        self.area = [0,0,0,0]

        # Initialization of the area map with objects
        # Each 0 is the number of objects found in the area
        self.area_obj = [0,0,0,0]
        
    def update_world(self, posX, posY):
        
        # 0 means not discovered
        # 1 means discovered

        #Check the position of the agent
        # To update the value of the world
        
        # North
        # If North not discovered
        if self.area[0] != 1:
            if posY >= 0.72 :
                # If West not discovered
                if self.area[2] != 1:
                    if posX <= -0.72:
                        self.world[0][0] = 1
                    if posX >= -0.72 and posX <= -0.44:
                        self.world[0][1] = 1
                    if posX >= -0.44 and posX <= -0.16:
                        self.world[0][2] = 1
                if posX >= -0.16 and posX <= 0.12:
                    self.world[0][3] = 1
                if posX >= 0.12 and posX <= 0.4:
                    self.world[0][4] = 1
                # If East not discovered
                if self.area[3] != 1:
                    if posX >= 0.4 and posX <= 0.68:
                        self.world[0][5] = 1
                    if posX >= 0.68 and posX <= 0.96:
                        self.world[0][6] = 1
                    if posX >= 0.72:
                        self.world[0][7] = 1

            if posY >= 0.44 and posY <= 0.72 :
                # If West not discovered
                if self.area[2] != 1:
                    if posX <= -0.72:
                        self.world[1][0] = 1
                    if posX >= -0.72 and posX <= -0.44:
                        self.world[1][1] = 1
                    if posX >= -0.44 and posX <= -0.16:
                        self.world[1][2] = 1
                if posX >= -0.16 and posX <= 0.12:
                    self.world[1][3] = 1
                if posX >= 0.12 and posX <= 0.4:
                    self.world[1][4] = 1
                # If East not discovered
                if self.area[3] != 1:
                    if posX >= 0.4 and posX <= 0.68:
                        self.world[1][5] = 1
                    if posX >= 0.68 and posX <= 0.96:
                        self.world[1][6] = 1
                    if posX >= 0.72:
                        self.world[1][7] = 1

            if posY >= 0.16 and posY <= 0.44 :
                # If West not discovered
                if self.area[2] != 1:
                    if posX <= -0.72:
                        self.world[2][0] = 1
                    if posX >= -0.72 and posX <= -0.44:
                        self.world[2][1] = 1
                    if posX >= -0.44 and posX <= -0.16:
                        self.world[2][2] = 1
                if posX >= -0.16 and posX <= 0.12:
                    self.world[2][3] = 1
                if posX >= 0.12 and posX <= 0.4:
                    self.world[2][4] = 1
                # If East not discovered
                if self.area[3] != 1:
                    if posX >= 0.4 and posX <= 0.68:
                        self.world[2][5] = 1
                    if posX >= 0.68 and posX <= 0.96:
                        self.world[2][6] = 1
                    if posX >= 0.72:
                        self.world[2][7] = 1

        # Center
        if posY >= -0.12 and posY <= 0.16 :
            # If West not discovered
            if self.area[2] != 1:
                if posX <= -0.72:
                    self.world[3][0] = 1
                if posX >= -0.72 and posX <= -0.44:
                    self.world[3][1] = 1
                if posX >= -0.44 and posX <= -0.16:
                    self.world[3][2] = 1
            if posX >= -0.16 and posX <= 0.12:
                self.world[3][3] = 1
            if posX >= 0.12 and posX <= 0.4:
                self.world[3][4] = 1
            # If East not discovered
            if self.area[3] != 1:
                if posX >= 0.4 and posX <= 0.68:
                    self.world[3][5] = 1
                if posX >= 0.68 and posX <= 0.96:
                    self.world[3][6] = 1
                if posX >= 0.72:
                    self.world[3][7] = 1
        
        # South
        if self.area[1] != 1:
            if posY >= -0.4 and posY <= -0.12 :
                # If West not discovered
                if self.area[2] != 1:
                    if posX <= -0.72:
                        self.world[4][0] = 1
                    if posX >= -0.72 and posX <= -0.44:
                        self.world[4][1] = 1
                    if posX >= -0.44 and posX <= -0.16:
                        self.world[4][2] = 1
                if posX >= -0.16 and posX <= 0.12:
                    self.world[4][3] = 1
                if posX >= 0.12 and posX <= 0.4:
                    self.world[4][4] = 1
                # If East not discovered
                if self.area[3] != 1:
                    if posX >= 0.4 and posX <= 0.68:
                        self.world[4][5] = 1
                    if posX >= 0.68 and posX <= 0.96:
                        self.world[4][6] = 1
                    if posX >= 0.72:
                        self.world[4][7] = 1
        
        
            if posY >= -0.68 and posY <= -0.4 :
                # If West not discovered
                if self.area[2] != 1:
                    if posX <= -0.72:
                        self.world[5][0] = 1
                    if posX >= -0.72 and posX <= -0.44:
                        self.world[5][1] = 1
                    if posX >= -0.44 and posX <= -0.16:
                        self.world[5][2] = 1
                if posX >= -0.16 and posX <= 0.12:
                    self.world[5][3] = 1
                if posX >= 0.12 and posX <= 0.4:
                    self.world[5][4] = 1
                # If East not discovered
                if self.area[3] != 1:
                    if posX >= 0.4 and posX <= 0.68:
                        self.world[5][5] = 1
                    if posX >= 0.68 and posX <= 0.96:
                        self.world[5][6] = 1
                    if posX >= 0.72:
                        self.world[5][7] = 1

            if posY >= -0.96 and posY <= -0.68 :
                # If West not discovered
                if self.area[2] != 1:
                    if posX <= -0.72:
                        self.world[6][0] = 1
                    if posX >= -0.72 and posX <= -0.44:
                        self.world[6][1] = 1
                    if posX >= -0.44 and posX <= -0.16:
                        self.world[6][2] = 1
                if posX >= -0.16 and posX <= 0.12:
                    self.world[6][3] = 1
                if posX >= 0.12 and posX <= 0.4:
                    self.world[6][4] = 1
                # If East not discovered
                if self.area[3] != 1:
                    if posX >= 0.4 and posX <= 0.68:
                        self.world[6][5] = 1
                    if posX >= 0.68 and posX <= 0.96:
                        self.world[6][6] = 1
                    if posX >= 0.72:
                        self.world[6][7] = 1

            if posY <= -0.72 :
                # If West not discovered
                if self.area[2] != 1:
                    if posX <= -0.72:
                        self.world[7][0] = 1
                    if posX >= -0.72 and posX <= -0.44:
                        self.world[7][1] = 1
                    if posX >= -0.44 and posX <= -0.16:
                        self.world[7][2] = 1
                if posX >= -0.16 and posX <= 0.12:
                    self.world[7][3] = 1
                if posX >= 0.12 and posX <= 0.4:
                    self.world[7][4] = 1
                # If East not discovered
                if self.area[3] != 1:
                    if posX >= 0.4 and posX <= 0.68:
                        self.world[7][5] = 1
                    if posX >= 0.68 and posX <= 0.96:
                        self.world[7][6] = 1
                    if posX >= 0.72:
                        self.world[7][7] = 1
        
        for l in range(8) :   
            print(self.world[l])

    def update_area(self):
        # Check the world to see if some area were fully discovered

        # If North is not fully discovered
        if self.area[0] != 1:
            print("check N")
            count = 0
            for i in range(3) :
                for j in range(8):
                    if self.world[i][j] == 1 :
                        count = count + 1
                    else:
                        break
            if count == 24:
                print("All checked")
                if self.area[0] == 0:
                    self.area[0] = 1
                    return self.not_sentence(0)
                else:
                    return self.not_sentence(0)
        else :
            print("N")

        # If South is not fully discovered
        if self.area[1] != 1:
            print("check S")
            count = 0
            for i in range(4,8) :
                for j in range(8):
                    if self.world[i][j] == 1 :
                        count = count + 1
                    else:
                        break
            if count == 32:
                print("All checked")
                if self.area[1] == 0:
                    self.area[1] = 1
                    return self.not_sentence(1)
                else:
                    return self.not_sentence(1)
        else :
            print("S")

        # If West is not fully discovered
        if self.area[2] != 1:
            print("check W")
            count = 0
            for i in range(3) :
                for j in range(8):
                    if self.world[j][i] == 1 :
                        count = count + 1
                    else:
                        break
            if count == 24:
                print("All checked")
                if self.area[2] == 0:
                    self.area[2] = 1
                    return self.not_sentence(2)
                else:
                    return  self.not_sentence(2)
        else :
            print("W")

        # If East is not fully discovered
        if self.area[3] != 1:
            print("check E")
            count = 0
            for i in range(4,8) :
                for j in range(8):
                    if self.world[j][i] == 1 :
                        count = count + 1
                    else:
                        break
            if count == 32:
                print("All checked")
                if self.area[3] == 0:
                    self.area[3] = 1
                    return self.not_sentence(3)
                else:
                    return self.not_sentence(3)
        else :
            print("E")

    def reset_area(self, num):

        self.area[num] = 0
        # If the area was North
        if num == 0:
            for i in range(3) :
                for j in range(8):
                    self.world[i][j] = 0 

        # If the area was South
        if num == 1:
            for i in range(4,8) :
                for j in range(8):
                    self.world[i][j] = 0 

        # If the area was West
        if num == 2:
            for i in range(3) :
                for j in range(8):
                    self.world[j][i] = 0 

        # If the area was East
        if num == 3:
            for i in range(4,8) :
                for j in range(8):
                    self.world[j][i] = 0 



    def not_sentence(self, i):
        print("------------- NOT SENTENCE ------------")
        print(self.area)
        print(str(self.area[i]))
        if self.area[i] == 1:
            self.reset_area(i)
            if i == 0 :
                return ["Object","Not","North","Landmark","Not","North"]
            if i == 1 :
                return ["Object","Not","South","Landmark","Not","South"]
            if i == 2 :
                return ["Object","Not","West","Landmark","Not","West"]
            if i == 3 :
                return ["Object","Not","East","Landmark","Not","East"]
        elif self.area[i] == 3 :
            self.reset_area(i)
            if i == 0 :
                return ["Object","Not","North"]
            if i == 1 :
                return ["Object","Not","South"]
            if i == 2 :
                return ["Object","Not","West"]
            if i == 3 :
                return ["Object","Not","East"]
        elif self.area[i] == 2 :
            self.reset_area(i)
            if i == 0 :
                return ["Landmark","Not","North"]
            if i == 1 :
                return ["Landmark","Not","South"]
            if i == 2 :
                return ["Landmark","Not","West"]
            if i == 3 :
                return ["Landmark","Not","East"]
        # Otherwise it == 4 and means the 2 objects are there 
        else: 
            self.reset_area(i)


    def update_area_obj(self, agent_x, agent_y, num):
        # Num : 2 if object
        #       3 if landmark
        # North / South
        if agent_y >= 0.32:
            if self.area[0] != 0:
                self.area[0] = 4
            else:
                self.area[0] = num
        if agent_y < -0.32:
            if self.area[1] != 0:
                self.area[1] = 4
            else:
                self.area[1] = num
        
        # East / West
        if agent_x >= 0.32:
            if self.area[3] != 0:
                self.area[3] = 4
            else:
                self.area[3] = num
        if agent_x < -0.32:
            if self.area[2] != 0:
                self.area[2] = 4
            else:
                self.area[2] = num

        pass

    def update_direction(self, direction):
        # Check if the current direction of the agent
        # Is the same as the old direction
        if self.directions == direction:
            # Increment time by 1
            self.time = self.time + 1
        # Or reset the direction and time
        else:
            self.directions = direction
            self.time = 0
        
        # If the agent is going in the same direction for a long time
        if self.time >= 2 and self.directions != [] :
            return True
        else:
            return False

    def parse_obs_strat(self, obs, sce_conf):
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []
        # If the action of pushing happens
        push = False
        
        # Direction of the agent
        direction = []


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
                # If collision with self
                # Don't print anything about the position
                if collision :
                    sentence.pop()


                # Is it pushing an object
                for object in range(int(sce_conf['nb_objects'])):
                    # Calculate the place in the array
                    spot = 4 # 4 values of the self agent
                    # 5 values for each agents (not self)
                    spot = spot + (int(sce_conf['nb_agents'])-1)*5 
                    # 5 values for each other objects
                    spot = spot + object*5 

                    # If visible                                      
                    if obs[0][spot] == 1 :

                        # Is it pushing ?
                        # Calculate the distance of the center 
                        # Of the object from the agent
                        x = obs[0][place+1] - obs[0][spot+1]
                        y = obs[0][place+2] - obs[0][spot+2]
                        distance = x*x + y*y
                        distance = sqrt(distance)
                        
                        # If collision
                        if distance < 0.47:
                            sentence.extend(["You","Push","Object"])
                            push = True
                            # Calculate where the object was pushed 
                            # Based on its distance from the agent
                            if y > 0.20 and y < 0.50 :
                                sentence.append("South")
                            if y < -0.20 and y > -0.50 :
                                sentence.append("North")
                            if x > 0.20 and x < 0.50 :
                                sentence.append("West")
                            if x < -0.20 and x > -0.50:
                                sentence.append("East")
                
                # SEARCH FOR OTHER AGENTS
                # Is it moving but not pushing
                """if not push:
                    sentence.extend(["You","Search"])
                    search = False
                    if obs[0][place+4] > 0.5:
                        sentence.append("North")
                        search = True
                    if obs[0][place+4] < -0.5:
                        sentence.append("South")
                        search = True
                    if obs[0][place+3] > 0.5:
                        sentence.append("East")
                        search = True
                    if obs[0][place+3] < -0.5:
                        sentence.append("West")
                        search = True
                    # If the agent is not moving
                    # Remove the beginning of the sentence
                    if not search:
                        sentence.pop()
                        sentence.pop()"""


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

                #We update the area_obj
                self.update_area_obj(obs[0][0],obs[0][1],2)

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
                distance = obs[0][place+1]*obs[0][place+1] + \
                    obs[0][place+2]*obs[0][place+2]
                distance = sqrt(distance)
    

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

                #We update the area_obj
                self.update_area_obj(obs[0][0],obs[0][1],3)

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
        # Set the direction vector depending on the direction of the agent 
        if obs[0][3] > 0.5:
            direction.append("North")
        if obs[0][3] < -0.5:
            direction.append("South")
        if obs[0][2] > 0.5:
            direction.append("East")
        if obs[0][2] < -0.5:
            direction.append("West")
        
        # Check if it had the same direction for a long time
        if self.update_direction(direction):
            # If not pushing generate the sentence
            # Depending on the speed of the agent
            if not push:
                sentence.extend(["I","Search"])
                if obs[0][3] > 0.5:
                    sentence.append("North")
                if obs[0][3] < -0.5:
                    sentence.append("South")
                if obs[0][2] > 0.5:
                    sentence.append("East")
                if obs[0][2] < -0.5:
                    sentence.append("West")


        # Update the world variable to see what the agent
        # Has discovered (to generate not sentences)
        self.update_world(obs[0][0],obs[0][1])
        temp = self.update_area()
        if temp != None:
            print("not none")
            sentence.extend(temp)

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
    """agents = []
    for agent in range(sce_conf["nb_agents"]):
        agents.append(TrackingAgent())"""

    observation = ObservationParserStrat(args)
    # Save all the sentences generated
    #sentences = []
    
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
            #sentences.append(sentence)

            time.sleep(args.step_time)
            env.render()

            if dones[0]:
                break
            obs = next_obs
    
    #print(sentences)

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