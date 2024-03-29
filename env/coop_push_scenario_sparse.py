import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Action, Entity

# For parser
from utils.parsers import Parser
from utils.mapper import Mapper
import random
from math import sqrt


LANDMARK_SIZE = 0.1
OBJECT_SIZE = 0.15
OBJECT_MASS = 1.0
AGENT_SIZE = 0.04
AGENT_MASS = 0.4

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)

def obj_callback(agent, world):
    action = Action()
    action.u = np.zeros((world.dim_p))
    action.c = np.zeros((world.dim_c))
    return action

# --------- Parser --------- 
# Strat parser
class ObservationParserStrat(Parser):
    
    vocab = ['Located', 'Object', 'Landmark', 'I', 'You', 'North', 'South', 'East', 'West', 'Center', 'Not', 'Push', 'Search']

    def __init__(self, sce_conf, obj_colors, obj_shapes, land_colors, land_shapes):
        """
        ObservationParserStrat, generate sentences for the agents and share actions
        :param sce_conf: (dict) information on the scenario
        obj_colors: list(int) all the colors of the objects
        obj_shapes: list(int) all the shapes of the objects
        land_colors: list(int) all the colors of the landmarks
        land_shapes: list(int) all the shapes of the landmarks
        """
        self.nb_agents = sce_conf['nb_agents']
        self.nb_objects = sce_conf['nb_objects']
        self.directions = []
        for nb_agent in range(self.nb_agents):
            newAgent = []
            self.directions.append(newAgent)
        self.time = []
        for nb_agent in range(self.nb_agents):
            self.time.append(0)

        self.map = Mapper(sce_conf)

    # Generate a not sentence depending on what the agent saw
    def not_sentence(self, i, j, nb_agent):
        '''
        Might Create a "Not sentence" if the agent don't see 1 
        or more types of object

        Input:  
            i and j:    (int) area_nb to check i could be refered as x and j as y
            nb_agent:   (int) nb of the agent

        Output: A sentence based on what it didn't see
        '''
        # Position of the agent
        position = []
        # Part of the sentence generated
        n_sent = []
        # Variable to check if you need to verify a whole area
        check = ""
        # Variable to check if the agent saw objects
        obj = 0

        # Depending on the position of the agent
        # Set the "position" variable
        if i == 0 :
            position.append("North")
            if j == 1 :
                check="North"
        if i == 2:
            position.append("South")
            if j == 1 :
                check="South"
        if j == 0:
            position.append("West")
            if i == 1 :
                check="West"
        if j == 2:
            position.append("East")
            if i == 1 :
                check="East"
        if i == 1 and j == 1:
            position.append("Center")
            check="Center"

        # Initalize the obj variable
        obj = self.map.area_obj[nb_agent][i][j]
        if obj >=4 and obj != 5:
            obj = obj//2
            
        # If the agent discovered a whole area
        # We have to check all 3 areas
        if check == "North":
            # Update the object value depending on the objects in the area
            obj = self.map.check_area(nb_agent, 0, 0, obj)
            # Then we reset the area
            self.map.reset_areas(0, nb_agent)
        elif check == "South":
            obj = self.map.check_area(nb_agent, 0, 2, obj)
            self.map.reset_areas(1, nb_agent)
        elif check == "West":
            obj = self.map.check_area(nb_agent, 1, 0, obj)
            self.map.reset_areas(2, nb_agent)
        elif check == "East":
            obj = self.map.check_area(nb_agent, 1, 2, obj)
            self.map.reset_areas(3, nb_agent)
        elif check == "Center":
            # Always reset the center
            self.map.reset_areas(4, nb_agent)

        # Depending on the objects in the area
        # Creates the not sentence
        if obj == 0:
            n_sent.extend(["Object","Not"])
            n_sent.extend(position)
            n_sent.extend(["Landmark","Not"])
            n_sent.extend(position)
        if obj == 2:
            n_sent.extend(["Landmark","Not"])
            n_sent.extend(position)
        if obj == 3:
            n_sent.extend(["Object","Not"])
            n_sent.extend(position)
        # If obj == 4 : both object are in the area

        return n_sent 

    # Update the agents's direction
    def update_direction(self, obs, nb_agent):
        '''
        Check if the current direction is the same as before 
        And return True if the agent has been going in the same direction
        for a long time (here more than 2 steps)

        Input:  
            obs: list(float) The position of the agent
                             (2 values for position, 2 values for velocity)
            nb_agent: (int) number of the agent

        Output: True or False
        '''

        direction = []

        # Search
        # Set the direction vector depending on the direction of the agent 
        if  obs[3] > 0.5:
            direction.append("North")
        if  obs[3] < -0.5:
            direction.append("South")
        if  obs[2] > 0.5:
            direction.append("East")
        if  obs[2] < -0.5:
            direction.append("West")

        # Check if the current direction of the agent
        # Is the same as the old direction
        if self.directions[nb_agent] == direction:
            # Increment time by 1
            self.time[nb_agent] = self.time[nb_agent] + 1
        # Or reset the direction and time
        else:
            self.directions[nb_agent] = direction
            self.time[nb_agent] = 0
        
        # If the agent is going in the same direction for a long time
        if self.time[nb_agent] >= 2 and self.directions[nb_agent] != [] :
            return True
        else:
            return False

    # Generate a sentence if the agent is "searching"
    def search_sentence(self, obs, nb_agent, push):
        '''
        Generate a sentence if the agent has been going in the same
        direction for a while

        Input:  
            obs: list(float) The position of the agent 
                             (2 values for position, 2 values for velocity)
            nb_agent: (int) Number of the agent
            push: (bool) We don't generate a search sentence if the agent is pushing

        Output: list(str) The search sentence generated
        '''
        sentence = []
        
        # Check if it had the same direction for a long time
        if self.update_direction(obs, nb_agent):
            # If not pushing generate the sentence
            # Depending on the speed of the agent
            if not push:
                sentence.extend(["I","Search"])
                if  obs[3] > 0.5:
                    sentence.append("North")
                if  obs[3] < -0.5:
                    sentence.append("South")
                if  obs[2] > 0.5:
                    sentence.append("East")
                if  obs[2] < -0.5:
                    sentence.append("West")

        return sentence

    # Generate a sentence if we see another agent
    def agent_sentence(self, obs, agent_i):
        '''
        If the agent sees another agent, it will generate a sentence
        with the position of this other agent (depending on the main agent)
        and will generate another sentence if this agent is pushing and object

        Input:  
            obs: list(float) All the observation starting at this agent
            agent_i: (int) Number of the agent

        Output: list(str) The agent sentence generated
        '''
        sentence = []

        # If visible                                      
        if  obs[0] == 1 :
            # Position
            sentence.append("You")
            collision = True

            # North / South
            if  obs[2] >= 0.15:
                sentence.append("North")
                collision = False
            elif  obs[2] < -0.15:
                sentence.append("South")
                collision = False
            
            # West / East
            if  obs[1] >= 0.15:
                sentence.append("East")
                collision = False
            elif  obs[1] < -0.15:
                sentence.append("West")
                collision = False
            # If collision with self
            # Don't print anything about the position
            if collision :
                sentence.pop()


            # Is it pushing an object
            for object in range(int(self.nb_objects)):
                # 5 values for each agents (not self)
                spot = (int(self.nb_agents)-agent_i-1)*5 
                # 5 values for each other objects
                spot = spot + object*5 

                # If visible                                      
                if  obs[spot] == 1 :

                    # Is it pushing ?
                    # Calculate the distance of the center 
                    # Of the object from the agent
                    x =  obs[1] -  obs[spot+1]
                    y =  obs[2] -  obs[spot+2]
                    distance = x*x + y*y
                    distance = sqrt(distance)
                    
                    # If collision
                    if distance < 0.47:
                        sentence.extend(["You","Push","Object"])
                        # Calculate where the object was pushed 
                        # Based on its distance from the agent
                        if y > 0.20 and y < 0.50 :
                            sentence.append("South")
                        elif y < -0.20 and y > -0.50 :
                            sentence.append("North")
                        if x > 0.20 and x < 0.50 :
                            sentence.append("West")
                        elif x < -0.20 and x > -0.50:
                            sentence.append("East")
                
        return sentence

    # Generate a sentence for the object
    def object_sentence(self, pos_agent, obs, nb_agent):
        '''
        Will generate a sentence if the agent sees the object
        with the position of this object and will generate another sentence
        if this agent is pushing the object

        Input:  
            pos_agent: list(float) position x and y of the agent
            obs: list(float) observation link to this object (1: visible or not, 2: position, 2: velocity)
            nb_agent: (int) Number of the agent

        Output: sentence: list(str) The object sentence generated
                push: (bool) True if the agent is pushing an object
        '''

        sentence = []
        # Check if the agent is pushing an object
        push = False

        # If visible                                      
        if  obs[0] == 1 :

            #We update the area_obj
            self.map.update_area_obj(pos_agent[0], pos_agent[1],2,nb_agent)

            sentence.append("Object")
            # North / South
            if  obs[2] >= 0.25:
                sentence.append("North")
            elif  obs[2] < -0.25:
                sentence.append("South")
            
            # West / East
            if  obs[1] >= 0.25:
                sentence.append("East")
            elif  obs[1] < -0.25:
                sentence.append("West")

            # Calculate the distance of the center 
            # Of the object from the agent
            distance =  obs[1]* obs[1] + \
                    obs[2]* obs[2]
            distance = sqrt(distance)
    

            # If collision
            if distance < 0.47:
                sentence.extend(["I","Push","Object"])
                push = True
                # Calculate where the object was pushed 
                # Based on its distance from the agent
                if  obs[2] > 0.20 and  obs[2] < 0.50 :
                    sentence.append("North")
                if  obs[2] < -0.20 and  obs[2] > -0.50 :
                    sentence.append("South")
                if  obs[1] > 0.20 and  obs[1] < 0.50 :
                    sentence.append("East")
                if  obs[1] < -0.20 and  obs[1] > -0.50:
                    sentence.append("West")

        return sentence, push

    # Generate a sentence for the landmark
    def landmark_sentence(self, pos_agent, obs, nb_agent):
        '''
        Will generate a sentence if the agent sees the landmark
        with the position of this landmark

        Input:  
            pos_agent: list(float) position x and y of the agent
            obs: list(float) observation link to this object (1: visible or not, 2: position)
            nb_agent: (int) Number of the agent

        Output: sentence: list(str) The landmark sentence generated
        '''
        sentence = []
        # Variable to check if we are close to the landmark
        # = True until we check its position
        close = True

        # If visible
        if  obs[0] == 1 :

            #We update the area_obj
            self.map.update_area_obj(pos_agent[0], pos_agent[1],3, nb_agent)

            sentence.append("Landmark")
            
            # North / South
            if  obs[2] >= 0.2:
                sentence.append("North")
                close = False
            elif  obs[2] < -0.2:
                sentence.append("South")
                close = False

            # West / East
            if  obs[1] >= 0.2:
                sentence.append("East")
                close = False
            elif  obs[1] < -0.2:
                sentence.append("West")
                close = False
            
            #If we are close to landmark
            if close:
                # North / South
                if  obs[2] >= 0:
                    sentence.append("North")
                elif  obs[2] < 0:
                    sentence.append("South")
                    
                # West / East
                if  obs[1] >= 0:
                    sentence.append("East")
                elif  obs[1] < 0:
                    sentence.append("West")

        return sentence

    # Generate the full sentence for the agent
    def parse_obs(self, obs, nb_agent):
        '''
        Generate the full sentence for the agent

        Input:  
            obs: list(float) observation of the agent with all the entities
            nb_agent: (int) Number of the agent

        Output: sentence: list(str) The sentence generated
        '''
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []
        # If the action of pushing happens
        push = False

        # Get the position of the agent
        sentence.extend(self.position_agent(obs[0:2]))
        for i in range(1,len(sentence)):
            position.append(sentence[i])
        pos_agent = obs[0:2]

        # Will calculate the place in the array of each entity
        # There are 4 values for the main agent
        """
        2: position
        2: velocity
        """
        place = 4

        # Add the values of the other agents
        """
        1: visible or not
        2: position
        2: velocity
        """
        for agent_i in range(self.nb_agents - 1):
            # Other agents sentence
            sentence.extend(self.agent_sentence(obs[place:],agent_i))
            place = place + 5
        
        # Objects sentence
        """
        1: visible or not
        2: position
        2: velocity
        """
        objects_sentence = []
        for object in range(self.nb_objects):
            obj_sent , push = self.object_sentence(pos_agent,obs[place:place+3],nb_agent)
            objects_sentence.extend(obj_sent)
            place = place + 5
        sentence.extend(objects_sentence)

        # Landmarks sentence
        """
        1: visible or not
        2: position
        """
        landmarks_sentence = []
        for landmark in range(self.nb_objects):
            landmarks_sentence.extend(self.landmark_sentence(pos_agent,obs[place:place+3],nb_agent))
            place = place + 3
        sentence.extend(landmarks_sentence)

        # Search sentence
        # Send the main agent values
        """
        2:  position
        2: velocity
        """
        sentence.extend(self.search_sentence(obs[0:4], nb_agent, push))

        # Update the world variable to see what the agent
        # Has discovered (to generate not sentences)
        self.map.update_world(obs[0], obs[1], nb_agent)
        # Check the aread and if an area if fully discovered
        # It will call not_sentence()
        temp = self.map.update_area(nb_agent)
        if temp != None:
            not_sent = self.not_sentence(temp[0], temp[1], temp[2])
            sentence.extend(not_sent)

        return sentence

    # Reset the informations on the agents
    def reset(self, obj_colors, obj_shapes, land_colors, land_shapes):
        '''
        Reset the map of the agent
        '''
        # Reset the map
        self.map.reset()

# Simple parser
class ObservationParser(Parser):
    
    vocab = ['Located', 'Object', 'Landmark', 'North', 'South', 'East', 'West', 'Center', 'Not']

    def __init__(self, args, sce_conf, obj_colors, obj_shapes, land_colors, land_shapes):
        """
        ObservationParser, generate sentences for the agents
        :param sce_conf: (dict) information on the scenario
        obj_colors: list(int) all the colors of the objects
        obj_shapes: list(int) all the shapes of the objects
        land_colors: list(int) all the colors of the landmarks
        land_shapes: list(int) all the shapes of the landmarks
        """
        self.args = args
        self.nb_agents = sce_conf['nb_agents']
        self.nb_objects = sce_conf['nb_objects']

    # Generate a sentence for the object
    def object_sentence(self, obs):
        '''
        Will generate a sentence if the agent sees the object
        with the position of this object and will generate another sentence
        if this agent is pushing the object

        Input:  
            obs: list(float) observation link to this object (1: visible or not, 2: position, 2: velocity)

        Output: sentence: list(str) The object sentence generated
        '''
        sentence = []

        # If visible                                      
        if  obs[0] == 1 :
            sentence.append("Object")
            # North / South
            if  obs[2] >= 0.25:
                sentence.append("North")
            elif  obs[2] < -0.25:
                sentence.append("South")
            
            # West / East
            if  obs[1] >= 0.25:
                sentence.append("East")
            elif  obs[1] < -0.25:
                sentence.append("West")

        return sentence

     # Generate a sentence for the landmark
    def landmark_sentence(self, obs):
        '''
        Will generate a sentence if the agent sees the landmark
        with the position of this landmark

        Input:  
            obs: list(float) observation link to this object (1: visible or not, 2: position)

        Output: sentence: list(str) The landmark sentence generated
        '''
        sentence = []
        # Variable to check if we are close to the landmark
        # = True until we check its position
        close = True

        # If visible
        if  obs[0] == 1 :
            sentence.append("Landmark")
            
            # North / South
            if  obs[2] >= 0.2:
                sentence.append("North")
                close = False
            elif  obs[2] < -0.2:
                sentence.append("South")
                close = False

            # West / East
            if  obs[1] >= 0.2:
                sentence.append("East")
                close = False
            elif  obs[1] < -0.2:
                sentence.append("West")
                close = False
            
            #If we are close to landmark
            if close:
                # North / South
                if  obs[2] >= 0:
                    sentence.append("North")
                elif  obs[2] < 0:
                    sentence.append("South")
                    
                # West / East
                if  obs[1] >= 0:
                    sentence.append("East")
                elif  obs[1] < 0:
                    sentence.append("West")

        return sentence

    # Might generate a not sentence
    def not_sentence(self, position, no_objects, no_landmarks):
        '''
        Might Create a "Not sentence" if the agent don't see 1 
        or more types of object

        Input:  
            position: list(float) The position of the agent
            no_objects: (bool) True or False if we see an object
            no_landmarks: (bool) True or False if we see a landmark

        Output: list(str) A sentence based on what it doesn't see
        '''
        sentence = []

        #Generation of a NOT sentence ?
        """
        if = 1: Will generate not_sentence only for objects
        if = 2: Will generate not_sentence only for landmarks
        if = 3: Will generate not_sentence for both objects and landmarks
        """
        not_sentence = 0
        # We don't always generate not sentence
        if random.random() <= self.args.chance_not_sent:
            not_sentence = random.randint(1,3)

            if not_sentence == 1 and no_objects:
                # Object not sentence
                sentence.extend(["Object","Not"])
                for word in position:
                    sentence.append(word)
            elif not_sentence == 2 and no_landmarks:
                # Landmark not sentence
                sentence.extend(["Landmark","Not"])
                for word in position:
                    sentence.append(word)
            elif not_sentence == 3:
                # Both object
                if no_objects:
                    sentence.extend(["Object","Not"])
                    for word in position:
                        sentence.append(word)
                if no_landmarks:
                    sentence.extend(["Landmark","Not"])
                    for word in position:
                        sentence.append(word)

        return sentence
    
    # Generate the full sentence for the agent
    def parse_obs(self, obs):
        '''
        Generate the full sentence for the agent

        Input:  
            obs: list(float) observation of the agent with all the entities

        Output: sentence: list(str) The sentence generated
        '''
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []

        # Get the position of the agent
        sentence.extend(self.position_agent(obs[0:2]))
        for i in range(1,len(sentence)):
            position.append(sentence[i])

        # Will calculate the place in the array of each entity
        # There are 4 values for the main agent
        """
        2: position
        2: velocity
        """
        place = 4

        # Add the values of the other agents
        """
        1: visible or not
        2: position
        2: velocity
        """
        for agent_i in range(self.nb_agents - 1):
            place = place + 5
        
        # Objects sentence
        """
        1: visible or not
        2: position
        2: velocity
        """
        objects_sentence = []
        for object in range(self.nb_objects):
            objects_sentence.extend(self.object_sentence(obs[place:place+3]))
            place = place + 5
        no_objects = len(objects_sentence) == 0  
        sentence.extend(objects_sentence)

        # Landmarks sentence
        """
        1: visible or not
        2: position
        """
        landmarks_sentence = []
        for landmark in range(self.nb_objects):
            landmarks_sentence.extend(self.landmark_sentence(obs[place:place+3]))
            place = place + 3
        no_landmarks = len(landmarks_sentence) == 0
        sentence.extend(landmarks_sentence)

        # Not sentence
        sentence.extend(self.not_sentence(position, no_objects, no_landmarks))

        return sentence

    # Reset the informations on the agents
    def reset(self, obj_colors, obj_shapes, land_colors, land_shapes):
        # Nothing to reset
        pass

# --------- Scenario -----------
# All entities have a color and a shape
class Color_Shape_Entity(Entity):
    def __init__(self):
        super(Color_Shape_Entity, self).__init__()

        self.num_color = 0
        self.shape = "circle"
        self.num_shape = 0

# properties of landmark entities
class Landmark(Color_Shape_Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of object entities
class Object(Color_Shape_Entity):
    def __init__(self):
        super(Object, self).__init__()
        # Objects are movable
        self.movable = True

class PushWorld(World):
    def __init__(self, nb_agents, nb_objects):
        super(PushWorld, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(self.nb_agents)]
        # List of objects to push
        self.nb_objects = nb_objects
        self.objects = [Object() for _ in range(self.nb_objects)]
        # Corresponding landmarks
        self.landmarks = [Landmark() for _ in range(self.nb_objects)]
        # Distances between objects and their landmark
        self.obj_lm_dists = np.zeros(self.nb_objects)
        # Shaping reward based on distances between objects and lms
        self.shaping_reward = 0.0
        # Control inertia
        self.damping = 0.8

    @property
    def entities(self):
        return self.agents + self.objects + self.landmarks

    def reset(self):
        for i in range(self.nb_objects):
            self.init_object(i)

    def init_object(self, obj_i, min_dist=0.2, max_dist=1.5):
        # Random color for both entities
        color = np.random.uniform(0, 1, self.dim_color)
        # Object
        self.objects[obj_i].name = 'object %d' % len(self.objects)
        self.objects[obj_i].color = color
        self.objects[obj_i].size = OBJECT_SIZE
        self.objects[obj_i].initial_mass = OBJECT_MASS
        # Landmark
        self.landmarks[obj_i].name = 'landmark %d' % len(self.landmarks)
        self.landmarks[obj_i].collide = False
        self.landmarks[obj_i].color = color
        self.landmarks[obj_i].size = LANDMARK_SIZE
        # Set initial positions
        # # Fixed initial pos
        # self.objects[obj_i].state.p_pos = np.zeros(2)
        # self.landmarks[obj_i].state.p_pos = np.array([-0.5, -0.5])
        # return
        if min_dist is not None:
            while True:
                self.objects[obj_i].state.p_pos = np.random.uniform(
                    -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, self.dim_p)
                self.landmarks[obj_i].state.p_pos = np.random.uniform(
                    -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, self.dim_p)
                dist = get_dist(self.objects[obj_i].state.p_pos, 
                                self.landmarks[obj_i].state.p_pos)
                if dist > min_dist and dist < max_dist:
                    break
        else:
            dist = get_dist(self.objects[obj_i].state.p_pos, 
                            self.landmarks[obj_i].state.p_pos)
        # Set distances between objects and their landmark
        self.obj_lm_dists[obj_i] = dist

    def step(self):
        # s
        last_obj_lm_dists = np.copy(self.obj_lm_dists)
        super().step()
        # s'
        # Compute shaping reward
        self.shaping_reward = 0.0
        for obj_i in range(self.nb_objects):
            # Update dists
            self.obj_lm_dists[obj_i] = get_dist(
                self.objects[obj_i].state.p_pos,
                self.landmarks[obj_i].state.p_pos)
            # Compute reward
            self.shaping_reward += last_obj_lm_dists[obj_i] \
                                    - self.obj_lm_dists[obj_i]

    # Integrate state with walls blocking entities on each side
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) \
                        + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            # Check for wall collision
            temp_pos = entity.state.p_pos + entity.state.p_vel * self.dt
            # West wall
            if temp_pos[0] - entity.size < -1:
                entity.state.p_vel[0] = 0.0
                entity.state.p_pos[0] = -1.0 + entity.size
            # East wall
            if temp_pos[0] + entity.size > 1:
                entity.state.p_vel[0] = 0.0
                entity.state.p_pos[0] = 1.0 - entity.size
            # North wall
            if temp_pos[1] - entity.size < -1:
                entity.state.p_vel[1] = 0.0
                entity.state.p_pos[1] = -1.0 + entity.size
            # South wall
            if temp_pos[1] + entity.size > 1:
                entity.state.p_vel[1] = 0.0
                entity.state.p_pos[1] = 1.0 - entity.size
            entity.state.p_pos += entity.state.p_vel * self.dt
                

class Scenario(BaseScenario):

    def make_world(self, nb_agents=4, nb_objects=1, obs_range=0.4, nb_colors=1,
                   nb_shapes=1, collision_pen=1, relative_coord=True, dist_reward=False, 
                   reward_done=50, step_penalty=0.1, obj_lm_dist_range=[0.2, 1.5]):
        world = PushWorld(nb_agents, nb_objects)
        # add agent
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_SIZE
            agent.initial_mass = AGENT_MASS
            agent.color = np.array([0.0,0.0,0.0])
            agent.color[i % 3] = 1.0
        # Objects and landmarks
        self.nb_objects = nb_objects
        for i, object in enumerate(world.objects):
            # Random color for both entities
            color = np.random.uniform(0, 1, world.dim_color)
            object.name = 'object %d' % i
            object.color = color
            object.size = OBJECT_SIZE
            object.initial_mass = OBJECT_MASS
            # Corresponding Landmarks
            world.landmarks[i].name = 'landmark %d' % i
            world.landmarks[i].collide = False
            world.landmarks[i].color = color
            world.landmarks[i].size = LANDMARK_SIZE
        self.obj_lm_dist_range = obj_lm_dist_range
        # Scenario attributes
        self.obs_range = obs_range
        self.relative_coord = relative_coord
        self.dist_reward = dist_reward
        # Reward attributes
        self.collision_pen = collision_pen
        # Flag for end of episode
        self._done_flag = False
        # Reward for completing the task
        self.reward_done = reward_done
        # Penalty for step in the environment
        self.step_penalty = step_penalty
        # make initial conditions
        self.reset_world(world)

        return world

    def done(self, agent, world):
        # Done if all objects are on their landmarks
        return self._done_flag

    def reset_world(self, world, seed=None, init_pos=None):
        if seed is not None:
            np.random.seed(seed)
        # Check if init positions are valid
        if init_pos is not None:
            if (len(init_pos["agents"]) != self.nb_agents or 
                len(init_pos["objects"]) != self.nb_objects or
                len(init_pos["landmarks"]) != self.nb_objects):
                print("ERROR: The initial positions {} are not valid.".format(
                    init_pos))
                exit(1)
        # world.reset()
        # Agents' initial pos
        # # Fixed initial pos
        # world.agents[0].state.p_pos = np.array([0.5, -0.5])
        # world.agents[1].state.p_pos = np.array([-0.5, 0.5])
        for i, agent in enumerate(world.agents):
            if init_pos is None:
                agent.state.p_pos = np.random.uniform(
                    -1 + agent.size, 1 - agent.size, world.dim_p)
            else:
                agent.state.p_pos = np.array(init_pos["agents"][i])
            agent.state.c = np.zeros(world.dim_c)
        # Objects and landmarks' initial pos
        for i, object in enumerate(world.objects):
            if init_pos is None:
                while True:
                    object.state.p_pos = np.random.uniform(
                        -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, world.dim_p)
                    world.landmarks[i].state.p_pos = np.random.uniform(
                        -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, world.dim_p)
                    dist = get_dist(object.state.p_pos, 
                                    world.landmarks[i].state.p_pos)
                    if (self.obj_lm_dist_range is None  or 
                        (dist > self.obj_lm_dist_range[0] and 
                         dist < self.obj_lm_dist_range[1])):
                        break
            else:
                object.state.p_pos = np.array(init_pos["objects"][i])
                world.landmarks[i].state.p_pos = np.array(init_pos["landmarks"][i])
                dist = get_dist(object.state.p_pos, 
                                world.landmarks[i].state.p_pos)
            # Set distances between objects and their landmark
            world.obj_lm_dists[i] = dist
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)
        self._done_flag = False


    def reward(self, agent, world):
        # Reward = -1 x squared distance between objects and corresponding landmarks
        dists = [get_dist(obj.state.p_pos, 
                          world.landmarks[i].state.p_pos)
                    for i, obj in enumerate(world.objects)]
        # rew = -sum([pow(d * 3, 2) for d in dists])
        # rew = -sum(dists)
        # rew = -sum(np.exp(dists))
        # Shaped reward
        shaped = 100 * world.shaping_reward
        # if world.shaping_reward > 0:
        #     shaped = 100 * world.shaping_reward
        # else:
        #     shaped = 10 * world.shaping_reward
            # shaped = 0
        rew = -self.step_penalty + shaped

        # Reward if task complete
        self._done_flag = all(d <= LANDMARK_SIZE for d in dists)
        if self._done_flag:
            rew += self.reward_done

        # Penalty for collision between agents
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is agent: continue
                dist = get_dist(agent.state.p_pos, other_agent.state.p_pos)
                dist_min = agent.size + other_agent.size
                if dist <= dist_min:
                    # print("COLLISION")
                    rew -= self.collision_pen
        # Penalty for collision with wall
        # if (agent.state.p_pos - agent.size <= -1).any() or \
        #    (agent.state.p_pos + agent.size >= 1).any():
        #    rew -= self.collision_pen
        return rew

    def observation(self, agent, world):
        # Observation:
        #  - Agent state: position, velocity
        #  - Other agents and objects:
        #     - If in sight: [relative x, relative y, v_x, v_y]
        #     - If not: [0, 0, 0, 0]
        #  - Landmarks:
        #     - If in sight: [relative x, relative y]
        #     - If not: [0, 0]
        # => Full observation dim = 2 + 2 + 5 x (nb_agents_objects - 1) + 3 x (nb_landmarks)
        # All distances are divided by max_distance to be in [0, 1]
        entity_obs = []
        for entity in world.agents + world.objects:
            if entity is agent: continue
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                # Pos: relative normalised
                #entity_obs.append(np.concatenate((
                #    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range, entity.state.p_vel
                #)))
                # Pos: relative
                if self.relative_coord:
                    entity_obs.append(np.concatenate((
                        [1.0], # Bit saying entity is observed
                        (entity.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                        entity.state.p_vel # Velocity
                        # (entity.state.p_pos - agent.state.p_pos), entity.state.p_vel
                    )))
                # Pos: absolute
                else:
                    entity_obs.append(np.concatenate((
                        # [1.0], entity.state.p_pos, entity.state.p_vel
                        entity.state.p_pos, entity.state.p_vel
                    )))
            else:
                if self.relative_coord:
                    entity_obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
                else:
                    entity_obs.append(np.zeros(4))
        for entity in world.landmarks:
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                # Pos: relative normalised
                #entity_obs.append(np.concatenate((
                #    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range
                #)))
                # Pos: relative
                if self.relative_coord:
                    entity_obs.append(np.concatenate((
                        [1.0], 
                        (entity.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                    )))
                    # entity_obs.append(
                    #     entity.state.p_pos - agent.state.p_pos
                    # )
                # Pos: absolute
                else:
                    # entity_obs.append(np.concatenate((
                    #     [1.0], entity.state.p_pos
                    # )))
                    entity_obs.append(entity.state.p_pos)
            else:
                if self.relative_coord:
                    entity_obs.append(np.array([0.0, 1.0, 1.0]))
                else:
                    entity_obs.append(np.zeros(2))

        # Communication


        return np.concatenate([agent.state.p_pos, agent.state.p_vel] + entity_obs)