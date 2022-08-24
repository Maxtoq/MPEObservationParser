import numpy as np
from multiagent.core import World, Agent, Entity
from multiagent.scenario import BaseScenario

from utils.parsers import Parser
import random
from math import sqrt

LANDMARK_SIZE = 0.1
AGENT_SIZE = 0.04

# ---------- Parser ---------
class ObservationParser(Parser):
    
    vocab = ['Located', 'Object', 'Landmark', 'North', 'South', 'East', 'West', 'Center', 'Not', "Red", "Blue", "Yellow", "Green", "Black", "Purple", "Circle", "Square", "Triangle"]
    def __init__(self, args, obj_colors, obj_shapes, land_colors, land_shapes):
        self.args = args
        """self.colors = colors
        self.shapes = shapes"""

    #Check the position of the agent to see if it is in a corner
    def check_position(self, obs):
        if ( obs[0] >= 0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5) or
             obs[0] <= -0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5)):
            return True
        else:
            return False

    def landmarks_sentence(self, obs, sce_conf, not_sentence, position):

        sentence = []
        # All landmarks
        landmarks = [x for x in range(int(sce_conf['nb_agents']))]
        dist_min_land = 10
        dist_min_agent = 10
        land_place = 0
        land_name = 0


        # While True until we have generated a sentence
        while True:
            # Look for the closest landmark in landmarks
            # For each Landmark 
            for landmark in landmarks:
            #Calculate the place in the array
                place = 4 # 4 values of the self agent
                # 2 values for each landmark
                place = place + landmark*2

                # Calculate the distance of the center 
                # Of the landmark from the agent
                distance =  obs[place]* obs[place] + \
                    obs[place+1]* obs[place+1]
                distance = sqrt(distance)

                # Get the closest landmark
                if distance < dist_min_land:
                    dist_min_land = distance
                    land_place = place
                    land_name = landmark

            # Look if there is an agent on it
            for agent in range(int(sce_conf['nb_agents'])-1):
                # Calculate the place in the array
                spot = 4 # 4 values of the self agent
                # 2 values for each landmark
                # nb_landmark = nb_agent
                spot = spot + sce_conf['nb_agents']*2 
                # 2 values for each agent
                spot = spot + agent*2 

                # Is it close
                # Calculate the distance of the center 
                # Of the landmark from the agent
                x =  obs[land_place] -  obs[spot]
                y =  obs[land_place+1] -  obs[spot+1]
                distance = x*x + y*y
                distance = sqrt(distance)

                # Get the closest agent
                if distance < dist_min_agent:
                    dist_min_agent = distance
                    # ------------------- Agent pos ? If possition --------------------
                
            # If on landmark
            if dist_min_agent <= LANDMARK_SIZE:
                # ----------------------- ADD POSITION ??? -----------------------------
                sentence.extend(["Agent","On","Landmark " + str(land_name)])
                # Remove and re initialize the distance values
                landmarks.remove(land_name)
                dist_min_agent = 10
                dist_min_land = 10
            # If not on landmark
            else:
                # If on landmark
                if dist_min_land <= LANDMARK_SIZE:
                    sentence.extend(["I","On","Landmark " + str(land_name)])
                else:
                    # Return the sentence
                    # Generate the sentence
                    sentence.append("Landmark " + str(land_name))
                        
                    # North / South
                    if  obs[land_place+1] >= 0.2:
                        sentence.append("North")
                    if  obs[land_place+1] < -0.2:
                        sentence.append("South")
                        
                    # West / East
                    if  obs[land_place] >= 0.2:
                        sentence.append("East")
                    if  obs[land_place] < -0.2:
                        sentence.append("West")
                    
                    #If we are close to landmark
                    elif ( obs[land_place+1] < 0.2 and  obs[land_place+1] >= -0.2 and
                            obs[land_place] < 0.2 and  obs[land_place] >= -0.2):
                        # North / South
                        if  obs[land_place+1] >= 0:
                            sentence.append("North")
                        if  obs[land_place+1] < 0:
                            sentence.append("South")
                            
                        # West / East
                        if  obs[land_place] >= 0:
                            sentence.append("East")
                        if  obs[land_place] < 0:
                            sentence.append("West")

                    if dist_min_land < 0.5:
                        sentence.append("Close")

                break



                    


                
                


            


        """# For each Landmark 
        for landmark in range(int(sce_conf['nb_agents'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 2 values for each landmark
            place = place + landmark*2

            # Calculate the distance of the center 
            # Of the landmark from the agent
            distance =  obs[place]* obs[place] + \
                obs[place+1]* obs[place+1]
            distance = sqrt(distance)

            # Get the closest landmark
            if distance < dist_min:
                dist_min = distance
                land_place = place
                land_name = landmark



        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_agents'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 2 values for each landmark
            place = place + landmark*2

            # Calculate the distance of the center 
            # Of the landmark from the agent
            distance =  obs[place]* obs[place] + \
                obs[place+1]* obs[place+1]
            distance = sqrt(distance)

            # Get the closest landmark
            if distance < dist_min:
                dist_min = distance
                land_place = place
                land_name = landmark
        
        if dist_min <= LANDMARK_SIZE:
            sentence.extend(["I","On","Landmark " + str(land_name)])
        else:

            for agent in range(int(sce_conf['nb_agents'])):
                # Calculate the place in the array
                spot = 4 # 4 values of the self agent
                # 2 values for each landmark
                # nb_landmark = nb_agent
                spot = spot + sce_conf['nb_agents']*2 
                # 2 values for each agent
                spot = spot + agent*2 

                # Is it close
                # Calculate the distance of the center 
                # Of the object from the agent
                x =  obs[land_place] -  obs[spot]
                y =  obs[land_place+1] -  obs[spot+1]
                distance = x*x + y*y
                distance = sqrt(distance)
                
                # If on landmark
                if distance <= LANDMARK_SIZE:
                    sentence.extend(["Agent","On","Landmark " + str(land_name)])
                    break

-----------------------------------------------------------------

            # Generate the sentence
            sentence.append("Landmark " + str(land_name))
                
            # North / South
            if  obs[land_place+1] >= 0.2:
                sentence.append("North")
            if  obs[land_place+1] < -0.2:
                sentence.append("South")
                
            # West / East
            if  obs[land_place] >= 0.2:
                sentence.append("East")
            if  obs[land_place] < -0.2:
                sentence.append("West")
            
            #If we are close to landmark
            elif ( obs[land_place+1] < 0.2 and  obs[land_place+1] >= -0.2 and
                    obs[land_place] < 0.2 and  obs[land_place] >= -0.2):
                # North / South
                if  obs[land_place+1] >= 0:
                    sentence.append("North")
                if  obs[land_place+1] < 0:
                    sentence.append("South")
                    
                # West / East
                if  obs[land_place] >= 0:
                    sentence.append("East")
                if  obs[land_place] < 0:
                    sentence.append("West")

            if dist_min < 0.5:
                sentence.append("Close")"""

        return sentence

    def agents_sentence(self, obs, sce_conf):

        sentence = []
        dist_min = 10
        agent_place = 0

        # Position of the agent
        # For each agents 
        for agent in range(int(sce_conf['nb_agents'])-1):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 2 values for each landmark
            # nb_landmark = nb_agent
            place = place + sce_conf['nb_agents']*2 
            # 2 values for each agent
            place = place + agent*2 

            # Calculate the distance of the center 
            # Of the agent from the other agent
            distance =  obs[place]* obs[place] + \
                obs[place+1]* obs[place+1]
            distance = sqrt(distance)
            # Get the closest agent
            if distance < dist_min:
                dist_min = distance
                agent_place = place

                # Generate the sentence
        sentence.append("Agent")
            # North / South
        if  obs[agent_place+1] >= 0.2:
            sentence.append("North")
        if  obs[agent_place+1] < -0.2:
            sentence.append("South")
        
        # West / East
        if  obs[agent_place] >= 0.2:
            sentence.append("East")
        if  obs[agent_place] < -0.2:
            sentence.append("West")

        #If we are close to agent
        elif ( obs[agent_place+1] < 0.2 and  obs[agent_place+1] >= -0.2 and
                obs[agent_place] < 0.2 and  obs[agent_place] >= -0.2):
            # North / South
            if  obs[agent_place+1] >= 0:
                sentence.append("North")
            if  obs[agent_place+1] < 0:
                sentence.append("South")
                
            # West / East
            if  obs[agent_place] >= 0:
                sentence.append("East")
            if  obs[agent_place] < 0:
                sentence.append("West")

        if dist_min < 0.3:
            sentence.append("Close")
                
        return sentence

    def parse_obs(self, obs, sce_conf):
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []

        #Generation of a NOT sentence ?
        not_sentence = 0
        if random.random() <= self.args.chance_not_sent:
            not_sentence = random.randint(1,3)

        """# Get the position of the agent
        sentence.extend(self.position_agent(obs))
        for i in range(1,len(sentence)):
            position.append(sentence[i])"""

        # Other agents sentence
        sentence.extend(self.agents_sentence(obs, sce_conf))
        
        # Landmark sentence
        sentence.extend(self.landmarks_sentence(obs, sce_conf, \
                        not_sentence, position))

        return sentence

    def reset(self, sce_conf, obj_colors, obj_shapes, land_colors, land_shapes):
        # Reset the colors and shapes
        """self.colors = colors
        self.shapes = shapes"""
        pass



class Color_Shape_Entity(Entity):
    def __init__(self):
        super(Color_Shape_Entity, self).__init__()

        self.num_color = 0
        self.shape = "circle"
        self.num_shape = 0

    # Get the color based on the number
    def num_to_color(self, color):
        match color:
            # Red
            case 1:
                color = [1, 0.22745, 0.18431]
            # Blue
            case 2:
                color = [0, 0.38, 1]
            # Green
            case 3:
                color = [0.2, 0.78 , 0.35]
            # Yellow
            case 4:
                color = [1, 0.8 , 0]
            # Purple
            case 5:
                color = [0.8, 0.21, 0.98]
            #Black
            case 6:
                color = [0.3, 0.3, 0.3]

        return color

    # Get the color based on the number
    def color_to_num(self, color):
        match color:
            # Red
            case [1, 0.22745, 0.18431]:
                color = 1
            # Blue
            case [0, 0.38, 1]:
                color = 2
            # Green
            case [0.2, 0.78 , 0.35]:
                color = 3
            # Yellow
            case [1, 0.8 , 0]:
                color = 4
            # Purple
            case [0.8, 0.21, 0.98]:
                color = 5
            #Black
            case [0.3, 0.3, 0.3]:
                color = 6

        return color

    # Get the color based on the number
    def num_to_shape(self, shape):
        match shape:
            # Circle
            case 1:
                shape = "circle"
            # Square
            case 2:
                shape = "square"
            # Triangle
            case 3:
                shape = "triangle"

        return shape

    # Get the color based on the number
    def shape_to_num(self, shape):
        match shape:
            #Black
            case "circle":
                shape = 1
            # Red
            case "square":
                shape = 2
            # Blue
            case "triangle":
                shape = 3

        return shape

# properties of landmark entities
class Landmark(Color_Shape_Entity):
     def __init__(self):
        super(Landmark, self).__init__()

class Scenario(BaseScenario):
    def make_world(self, nb_agents=4, step_penalty=0.1):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = nb_agents
        num_landmarks = nb_agents
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = AGENT_SIZE = 0.04
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = LANDMARK_SIZE

        # Penalty for step in the environment
        self.step_penalty = step_penalty

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, seed=None, init_pos=None):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            print(landmark.state.p_pos)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        """# And a step penalty
        rew = -self.step_penalty"""
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            #comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        #return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
