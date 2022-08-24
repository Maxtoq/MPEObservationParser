import numpy as np
from multiagent.core import World, Agent, Entity
from multiagent.scenario import BaseScenario

from utils.parsers import Parser
import random
from math import sqrt

LANDMARK_SIZE = 0.1
AGENT_SIZE = 0.04

LANDMARK_COLORS = [
    [0.25,0.25,0.75], # Blue
    [0.25,0.75,0.25],# Green
    [0.75,0.25,0.25], # Red
    [0.1,0.1,0.1], # Black
    [0.85,0.85,0.0] # Yellow
]

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

        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_agents']+1)):
        #Calculate the place in the array
            place = 2 # 2 values of the self agent

            # 2 values for each landmark
            place = place + landmark*2

            sentence.append("Landmark")
            
            # North / South
            if  obs[place+1] >= 0.2:
                sentence.append("North")
            if  obs[place+1] < -0.2:
                sentence.append("South")
                
            # West / East
            if  obs[place] >= 0.2:
                sentence.append("East")
            if  obs[place] < -0.2:
                sentence.append("West")
            
            #If we are close to landmark
            elif ( obs[place+1] < 0.2 and  obs[place+1] >= -0.2 and
                    obs[place] < 0.2 and  obs[place] >= -0.2):
                # North / South
                if  obs[place+1] >= 0:
                    sentence.append("North")
                if  obs[place+1] < 0:
                    sentence.append("South")
                    
                # West / East
                if  obs[place] >= 0:
                    sentence.append("East")
                if  obs[place] < 0:
                    sentence.append("West")

            """# Calculate the distance of the center 
            # Of the landmark from the agent
            distance =  obs[place]* obs[place] + \
                obs[place+1]* obs[place+1]
            distance = sqrt(distance)

            if distance < 0.5:
               sentence.append("Close")
            elif  distance > 1.2:
                sentence.append("Far")"""

        return sentence

    def agents_sentence(self, obs, sce_conf):

        sentence = []

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

            # Position
            sentence.append("You")
                # North / South
            if  obs[place+1] >= 0.2:
                sentence.append("North")
            if  obs[place+1] < -0.2:
                sentence.append("South")
            
            # West / East
            if  obs[place] >= 0.2:
                sentence.append("East")
            if  obs[place] < -0.2:
                sentence.append("West")

            #If we are close to agent
            elif ( obs[place+1] < 0.2 and  obs[place+1] >= -0.2 and
                    obs[place] < 0.2 and  obs[place] >= -0.2):
                # North / South
                if  obs[place+1] >= 0:
                    sentence.append("North")
                if  obs[place+1] < 0:
                    sentence.append("South")
                    
                # West / East
                if  obs[place] >= 0:
                    sentence.append("East")
                if  obs[place] < 0:
                    sentence.append("West")

            # Calculate the distance of the center 
            # Of the agent from the other agent
            distance =  obs[place]* obs[place] + \
                obs[place+1]* obs[place+1]
            distance = sqrt(distance)

            if distance < 0.3:
               sentence.append("Close")
            elif  distance > 1.5:
                sentence.append("Far")
                
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

        """# Other agents sentence
        sentence.extend(self.agents_sentence(obs, sce_conf))"""
        
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
    def make_world(self, nb_agents=2, step_penalty=0.1):
        world = World()
        # set any world properties first
        world.dim_c = 10
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(nb_agents+1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, seed=None, init_pos=None):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        landmarks = random.sample(world.landmarks,len(world.agents) +1)
        for i in range(len(world.agents)-1):
            world.agents[i].goal_a = world.agents[i+1]
            world.agents[i].goal_b = landmarks.pop()
        world.agents[len(world.agents)-1].goal_a = world.agents[0]
        world.agents[len(world.agents)-1].goal_b = landmarks.pop()

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25]) 
        # random properties for landmarks
        colors = random.sample(LANDMARK_COLORS,len(world.agents) + 1)
        for i in range(len(world.agents)+1):
            world.landmarks[i].color = colors.pop()

        # special colors for goals
        for i in range(len(world.agents)):
            world.agents[i].goal_a.color = world.agents[i].goal_b.color                              
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color 

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)

        return np.concatenate([agent.state.p_vel] + entity_pos+ entity_color + [goal_color[1]])# + comm)
            