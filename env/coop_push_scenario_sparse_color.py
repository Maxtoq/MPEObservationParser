import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Action, Entity


# For parser
from utils.parsers import ColorParser
from utils.mapper import ColorMapper
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
# Color parsers
class ObservationParserStrat(ColorParser):
    
    vocab = ['Located', 'Object', 'Landmark', 'I', 'You', 'North', 'South', 'East', 'West', 'Center', 'Not', 'Push', 'Search', "Red", "Blue", "Yellow", "Green", "Black", "Purple"]

    def __init__(self, args, sce_conf, obj_colors, obj_shapes, land_colors, land_shapes):
        self.args = args
        self.directions = []
        for nb_agent in range(sce_conf['nb_agents']):
            newAgent = []
            self.directions.append(newAgent)
        self.time = []
        for nb_agent in range(sce_conf['nb_agents']):
            self.time.append(0)

        self.map = ColorMapper(args, sce_conf)
        self.obj_colors = obj_colors
        self.land_colors = land_colors

    # Return a random object based on the list of objects
    def select_not_object(self, objects):
        list_obj = []
        list_land = []
        list = []
        if len(objects) > 0:
            # Choose the type of sentence
            """
            1: return 1 landmark and 1 object
            2: return 1 object
            3: return 1 landmark
            """
            type = random.randint(1,3)
            if type == 1:
                for obj in objects:
                    if obj[0] == 2:
                        list_obj.append(obj)
                for obj in objects:
                    if obj[0] == 3:
                        list_land.append(obj)
                if len(list_obj) > 0 :
                    list.extend([random.choice(list_obj)])
                if len(list_land) > 0 :
                    list.extend([random.choice(list_land)])
            else :
                for obj in objects:
                    if obj[0] == type:
                        list_obj.append([obj])
                if len(list_obj) > 0 :
                    list.extend(random.choice(list_obj)) 

        return list

    def not_sentence(self, i, j, nb_agent):
        # Position of the agent
        position = []
        # Part of the sentence generated
        n_sent = []
        # Variable to check if you need to verify a whole area
        check = ""
        # Variable to check if the agent saw objects
        objects = []
        obj_name = {2: "Object", 3: "Landmark"}

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

        # If the agent discovered a whole area
        # We have to check all 3 areas
        if check == "North":
            # Update the object value depending on the objects in the area
            objects = self.map.check_area(nb_agent, 0, 0, self.obj_colors)
            # Then we reset the area
            self.map.reset_areas(0, nb_agent)
        elif check == "South":
            objects = self.map.check_area(nb_agent, 0, 2, self.obj_colors)
            self.map.reset_areas(1, nb_agent)
        elif check == "West":
            objects = self.map.check_area(nb_agent, 1, 0, self.obj_colors)
            self.map.reset_areas(2, nb_agent)
        elif check == "East":
            objects = self.map.check_area(nb_agent, 1, 2, self.obj_colors)
            self.map.reset_areas(3, nb_agent)
        elif check == "Center":
            # Always reset the center
            self.map.reset_areas(4, nb_agent)                
        # If we don't have to check a big area
        else :
            objects = self.map.find_missing(nb_agent, i, j, self.obj_colors)

        # Depending on the objects in the area
        # Creates the not sentence
        objects = self.select_not_object(objects)

        # One object sentence
        for i in range(len(objects)):
            n_sent.extend([self.get_color(objects[i][1]),obj_name[objects[i][0]], \
                "Not"])
            n_sent.extend(position)

        return n_sent 

    def update_direction(self, direction, nb_agent):
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

    def agents_sentence(self, obs, sce_conf):

        sentence = []

        # Position of the agent
        # For each agents 
        for agent in range(int(sce_conf['nb_agents'])-1):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents 
            place = place + agent*5 

            # If visible                                      
            if  obs[place] == 1 :

                # Position
                sentence.append("You")
                collision = True
                 # North / South
                if  obs[place+2] >= 0.15:
                    sentence.append("North")
                    collision = False
                if  obs[place+2] < -0.15:
                    sentence.append("South")
                    collision = False
                
                # West / East
                if  obs[place+1] >= 0.15:
                    sentence.append("East")
                    collision = False
                if  obs[place+1] < -0.15:
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
                    # 8 values for each other objects
                    spot = spot + object*8

                    # If visible                                      
                    if  obs[spot] == 1 :

                        # Is it pushing ?
                        # Calculate the distance of the center 
                        # Of the object from the agent
                        x =  obs[place+1] -  obs[spot+1]
                        y =  obs[place+2] -  obs[spot+2]
                        distance = x*x + y*y
                        distance = sqrt(distance)
                        
                        # If collision
                        if distance < 0.47:
                            sentence.extend(["You","Push",self.array_to_color(obs[spot+5:spot+8]),"Object"])
                            #push = True
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
                
        return sentence

    def objects_sentence(self, obs, sce_conf, nb_agent):

        sentence = []
        push = False
        obj = 0

        # Position of the objects
        # For each object 
        for object in range(int(sce_conf['nb_objects'])):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5 
            # 8 values for each other objects
            place = place + object*8

            # If visible                                      
            if  obs[place] == 1 :

                #We update the area_obj
                """
                obs[0] pos x of the agent
                obs[1] pos y of the agent
                object type : 2
                object color
                nb of the agent
                """
                self.map.update_area_obj(obs[0], obs[1],2,self.obj_colors[obj],nb_agent)
                sentence.append(self.array_to_color(obs[place+5:place+8]))
                sentence.append("Object")
                 # North / South
                if  obs[place+2] >= 0.25:
                    sentence.append("North")
                if  obs[place+2] < -0.25:
                    sentence.append("South")
                
                # West / East
                if  obs[place+1] >= 0.25:
                    sentence.append("East")
                if  obs[place+1] < -0.25:
                    sentence.append("West")

                # Calculate the distance of the center 
                # Of the object from the agent
                distance =  obs[place+1]* obs[place+1] + \
                     obs[place+2]* obs[place+2]
                distance = sqrt(distance)
    

                # If collision
                if distance < 0.47:
                    sentence.extend(["I","Push",self.array_to_color(obs[place+5:place+8]),"Object"])
                    push = True
                    # Calculate where the object was pushed 
                    # Based on its distance from the agent
                    if  obs[place+2] > 0.20 and  obs[place+2] < 0.50 :
                        sentence.append("North")
                    if  obs[place+2] < -0.20 and  obs[place+2] > -0.50 :
                        sentence.append("South")
                    if  obs[place+1] > 0.20 and  obs[place+1] < 0.50 :
                        sentence.append("East")
                    if  obs[place+1] < -0.20 and  obs[place+1] > -0.50:
                        sentence.append("West")
            obj += 1

        return sentence , push

    def landmarks_sentence(self, obs, sce_conf, nb_agent):

        sentence = []
        obj = 0

        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_objects'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5
            # 8 values for each objects 
            place = place + int(sce_conf['nb_objects'])*8 
            # 6 values for each landmark
            place = place + landmark*6

            # If visible
            if  obs[place] == 1 :

                #We update the area_obj
                """
                obs[0] pos x of the agent
                obs[1] pos y of the agent
                object type : 2
                object color
                nb of the agent
                """
                self.map.update_area_obj( obs[0], obs[1],3, self.land_colors[obj], nb_agent)
                sentence.append(self.array_to_color(obs[place+3:place+6]))
                sentence.append("Landmark")
                
                # North / South
                if  obs[place+2] >= 0.2:
                    sentence.append("North")
                if  obs[place+2] < -0.2:
                    sentence.append("South")
                    
                # West / East
                if  obs[place+1] >= 0.2:
                    sentence.append("East")
                if  obs[place+1] < -0.2:
                    sentence.append("West")
                
                #If we are close to landmark
                elif ( obs[place+2] < 0.2 and  obs[place+2] >= -0.2 and
                     obs[place+1] < 0.2 and  obs[place+1] >= -0.2):
                    # North / South
                    if  obs[place+2] >= 0:
                        sentence.append("North")
                    if  obs[place+2] < 0:
                        sentence.append("South")
                        
                    # West / East
                    if  obs[place+1] >= 0:
                        sentence.append("East")
                    if  obs[place+1] < 0:
                        sentence.append("West")
            obj += 1

        return sentence

    def search_sentence(self, obs, nb_agent, push):
        sentence = []

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
        
        # Check if it had the same direction for a long time
        if self.update_direction(direction, nb_agent):
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

    def parse_obs(self, obs, sce_conf, nb_agent):
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []
        # If the action of pushing happens
        push = False

        # Get the position of the agent
        sentence.extend(self.position_agent(obs))
        for i in range(1,len(sentence)):
            position.append(sentence[i])
        
        # Other agents sentence
        sentence.extend(self.agents_sentence(obs, sce_conf))

        # Objects sentence
        obj_sent , push = self.objects_sentence(obs, sce_conf, nb_agent)
        sentence.extend(obj_sent)
        
        # Landmark sentence
        sentence.extend(self.landmarks_sentence(obs, sce_conf, nb_agent))
        
        # Search sentence
        sentence.extend(self.search_sentence(obs, nb_agent, push))

        # Update the world variable to see what the agent
        # Has discovered (to generate not sentences)
        self.map.update_world( obs[0], obs[1], nb_agent)
        temp = self.map.update_area(nb_agent)
        if temp != None:
            not_sent = self.not_sentence(temp[0], temp[1], temp[2])
            sentence.extend(not_sent)

        return sentence

    def reset(self, sce_conf, obj_colors, obj_shapes, land_colors, land_shapes):
        # Reset the map and the colors
        self.map.reset(sce_conf)
        self.land_colors = land_colors
        self.obj_colors = obj_colors

class ObservationParser(ColorParser):
    
    vocab = ['Located', 'Object', 'Landmark', 'North', 'South', 'East', 'West', 'Center', 'Not', "Red", "Blue", "Yellow", "Green", "Black", "Purple"]
    def __init__(self, args, sce_conf, obj_colors, obj_shapes, land_colors, land_shapes):
        self.args = args
        self.obj_colors = obj_colors
        self.land_colors = land_colors
        self.nb_agents = sce_conf['nb_agents']
        self.nb_objects = sce_conf['nb_objects']

    # Generate a sentence for the object
    def object_sentence(self, obs):
        '''
        Will generate a sentence if the agent sees the object

        Input:  The observation from the agent of this object

        Output: A sentence based on what it sees
        '''
        sentence = []

        # If visible                                      
        if  obs[0] == 1 :
            sentence.append(self.array_to_color(obs[5:8]))
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
        Will generate a sentence if the agent sees the object

        Input:  The observation from the agent of this object

        Output: A sentence based on what it sees
        '''
        sentence = []
        # Variable to check if we are close to the landmark
        # = True until we check its position
        close = True

        # If visible
        if  obs[0] == 1 :
            sentence.append(self.array_to_color(obs[3:6]))
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
                if  obs[2] < 0:
                    sentence.append("South")
                    
                # West / East
                if  obs[1] >= 0:
                    sentence.append("East")
                if  obs[1] < 0:
                    sentence.append("West")

        return sentence

    # Might generate a not sentence
    def not_sentence(self, position, not_vis_obj, not_vis_land):
        '''
        Might Create a "Not sentence" if the agent don't see 1 
        or more types of object

        Input:  
            position: The position of the agent
            no_objects: list of not visible object
            no_landmarks: list of not visible landmark

        Output: A sentence based on what it doesn't see
                list(str)
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

            if not_sentence == 1 and len(not_vis_obj) != 0:
                # Object not sentence
                # Pick a random object
                obj = random.choice(not_vis_obj)
                sentence.extend([self.get_color(self.obj_colors[obj]),"Object","Not"])
                for word in position:
                    sentence.append(word)
            elif not_sentence == 2 and len(not_vis_land) != 0:
                # Landmark not sentence
                # Pick a random object
                obj = random.choice(not_vis_land)
                sentence.extend([self.get_color(self.land_colors[obj]),"Landmark","Not"])
                for word in position:
                    sentence.append(word)
            elif not_sentence == 3:
                # Both object
                if len(not_vis_obj) != 0:
                    # Pick a random object
                    obj = random.choice(not_vis_obj)
                    sentence.extend([self.get_color(self.obj_colors[obj]),"Object","Not"])
                    for word in position:
                        sentence.append(word)
                if len(not_vis_land) != 0:
                    # Pick a random object
                    obj = random.choice(not_vis_land)
                    sentence.extend([self.get_color(self.land_colors[obj]),"Landmark","Not"])
                    for word in position:
                        sentence.append(word)

        return sentence

    def parse_obs(self, obs):
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
        3: color
        """
        objects_sentence = []
        not_visible_obj = []
        for object in range(self.nb_objects):
            object_sentence = []
            object_sentence.extend(self.object_sentence(obs[place:place+8]))
            place = place + 8
            # If we don't see the object
            if len(object_sentence) == 0:
                # Get the number of the object
                not_visible_obj.append(object)
            # Else we append the sentence
            else:
                objects_sentence.extend(object_sentence)
        #no_objects = len(objects_sentence) == 0  
        sentence.extend(objects_sentence)

        # Landmarks sentence
        """
        1: visible or not
        2: position
        3: color
        """
        landmarks_sentence = []
        not_visible_land = []
        for landmark in range(self.nb_objects):
            landmark_sentence = []
            landmark_sentence.extend(self.landmark_sentence(obs[place:place+6]))
            place = place + 6
            # If we don't see the object
            if len(landmark_sentence) == 0:
                # Get the number of the object
                not_visible_land.append(landmark)
            # Else we append the sentence
            else:
                landmarks_sentence.extend(landmark_sentence)
        #no_landmarks = len(landmarks_sentence) == 0
        sentence.extend(landmarks_sentence)

        # Not sentence
        sentence.extend(self.not_sentence(position, not_visible_obj, not_visible_land))

        return sentence

    def reset(self, sce_conf, obj_colors, obj_shapes, land_colors, land_shapes):
        # Reset the colors
        self.obj_colors = obj_colors
        self.land_colors = land_colors

# --------- Scenario -----------
# All entities have a color and a shape
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
        #color = self.pick_color()
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
        self.nb_colors = nb_colors
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_SIZE
            agent.initial_mass = AGENT_MASS
            agent.color = np.array([0.0,0.0,0.0])
            agent.color[i % 3] = 1.0
        # Objects and landmarks
        self.nb_objects = nb_objects
        
        # Set list of colors
        all_colors = [1,2,3]
        colors = []
        # List of objects_name
        objects_name = []
        for i, object in enumerate(world.objects):
            # Pick a color that is not already taken
            # Take a random color
            if len(colors) < nb_colors:
                color = all_colors.pop(0)
            # If we already have the maximum nb of color
            # We pick one from the ones we already have
            else:
                color = random.choice(colors)
            colors.append(color)

            object.name = 'object %d' % i
            object.num_color = color
            object.color = object.num_to_color(color)
            object.size = OBJECT_SIZE
            object.initial_mass = OBJECT_MASS
            objects_name.append(object.name)

        for land in world.landmarks:
            land.collide = False
            land.size = LANDMARK_SIZE

            # Take a random color
            color = random.sample(colors,1).pop()
            colors.remove(color)

            # Corresponding Landmarks
            for i, object in enumerate(world.objects):
                if object.num_color == color and object.name in objects_name:
                    land.name = 'landmark %d' % i
                    land.num_color = color
                    land.color = land.num_to_color(color)
                    objects_name.remove(object.name)
                    break
        
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
        all_colors = [1,2,3]
        colors = []
        color_count = 3
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

                # Pick a color that is not already taken
                # Take a random color
                if len(colors) < self.nb_colors:
                    if len(colors) < 3:
                        choice = random.randint(0,color_count-1)
                        color = all_colors[choice]
                        color_count -= 1
                    else:
                        color = random.sample(all_colors,1).pop()
                    all_colors.remove(color)

                # If we already have the maximum nb of color
                # We pick one from the ones we already have
                else:
                    color = random.choice(colors)
                colors.append(color)

                # Landmark
                landmark = None
                # Set color
                object.num_color = color
                object.color = object.num_to_color(color)

                for land in world.landmarks:
                    # Check if the landmark number is the same as the object number
                    o = int(object.name.split()[-1])
                    l = int(land.name.split()[-1])
                    # If same landmark
                    if o == l:
                        land.num_color = color
                        land.color = land.num_to_color(color)
                        landmark = land
                        break

                while True:
                    object.state.p_pos = np.random.uniform(
                        -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, world.dim_p)
                    if landmark != None:
                        landmark.state.p_pos = np.random.uniform(
                        -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, world.dim_p)

                        dist = get_dist(object.state.p_pos, 
                                    landmark.state.p_pos)   
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
        dists = []
        for obj in world.objects:
            # Set the minimal distance high
            mini = 10
            for land in world.landmarks:
                # Find the closest landmark with the same color
                if obj.num_color == land.num_color:
                    dist = get_dist(obj.state.p_pos, land.state.p_pos)
                    if dist < mini:
                        mini = dist
            # Append the distance with the closest landmark from the same color
            dists.append(mini)

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
        for entity in world.agents:
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
        for entity in world.objects:
            # Create list of colors for the observation
            color = [0] * 3
            color[entity.num_color-1] = 1
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
                        entity.state.p_vel, # Velocity
                        color
                        # (entity.state.p_pos - agent.state.p_pos), entity.state.p_vel
                    )))
                # Pos: absolute
                else:
                    entity_obs.append(np.concatenate((
                        # [1.0], entity.state.p_pos, entity.state.p_vel
                        entity.state.p_pos, entity.state.p_vel, color
                    )))
            else:
                if self.relative_coord:
                    entity_obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0, 0, 0]))
                else:
                    entity_obs.append(np.zeros(7))
        for entity in world.landmarks:
            # Create list of colors for the observation
            color = [0] * 3
            color[entity.num_color-1] = 1
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
                        color
                    )))
                    # entity_obs.append(
                    #     entity.state.p_pos - agent.state.p_pos
                    # )
                # Pos: absolute
                else:
                    # entity_obs.append(np.concatenate((
                    #     [1.0], entity.state.p_pos
                    # )))
                    entity_obs.extend(entity.state.p_pos, color)
            else:
                if self.relative_coord:
                    entity_obs.append(np.array([0.0, 1.0, 1.0, 0, 0, 0]))
                else:
                    entity_obs.append(np.zeros(5))

        # Communication


        return np.concatenate([agent.state.p_pos, agent.state.p_vel] + entity_obs)