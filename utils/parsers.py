import random
from math import sqrt

from abc import ABC, abstractmethod

from utils.mapper import Mapper, ColorMapper

# Mother classes
class Parser(ABC):
    """ Abstract Parser """
    @abstractmethod
    def parse_obs(self, obs, sce_conf):
        """
        Returns a sentence generated based on the actions of the agent
        """
        raise NotImplementedError

    def position_agent(self, obs):
        sentence = []
        # Position of the agent (at all time)
        sentence.append("Located")
        
        # North / South
        if  obs[1] >= 0.33:
            sentence.append("North")
        if  obs[1] < -0.33:
            sentence.append("South")
        
        # West / East
        if  obs[0] >= 0.33:
            sentence.append("East")
        if  obs[0] < -0.33:
            sentence.append("West")
        
        # Center
        elif len(sentence) == 1:
            sentence.append("Center")

        return sentence

    @abstractmethod
    def objects_sentence(self, obs, sce_conf):
        """
        Returns a sentence generated based on the objects see or not by the agent
        """
        raise NotImplementedError

    @abstractmethod
    def landmarks_sentence(self, obs, sce_conf):
        """
        Returns a sentence generated based on the landmarks see or not by the agent
        """
        raise NotImplementedError

class ColorParser(Parser):
    """ Abstract Parser """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    # Get the color based on the number
    def get_color(self, color):
        match color:
            # Red
            case 1:
                color = "Red"
            # Blue
            case 2:
                color = "Bleu"
            # Green
            case 3:
                color = "Green"
            # Yellow
            case 4:
                color = "Yellow"
            # Purple
            case 5:
                color = "Purple"
            #Black
            case 6:
                color = "Black"

        return color
        
class ColorShapeParser(ColorParser):
    """ Abstract Parser """
    # Get the shape based on the number
    def get_shape(self, shape):
        match shape:
            #Black
            case 1:
                shape = "Circle"
            # Red
            case 2:
                shape = "Square"
            # Blue
            case 3:
                shape = "Triangle"

        return shape


# Normal parsers
class ObservationParserStrat(Parser):
    
    vocab = ['Located', 'Object', 'Landmark', 'I', 'You', 'North', 'South', 'East', 'West', 'Center', 'Not', 'Push', 'Search']

    def __init__(self, args, sce_conf):
        self.args = args
        self.directions = []
        for nb_agent in range(sce_conf['nb_agents']):
            newAgent = []
            self.directions.append(newAgent)
        self.time = []
        for nb_agent in range(sce_conf['nb_agents']):
            self.time.append(0)

        self.map = Mapper(args, sce_conf)

    def not_sentence(self, i, j, nb_agent):
        """print("------------------------- NOT SENTENCE -------------------------")
        print(self.map.area[nb_agent])
        print(self.map.area_obj)"""
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
                    # 5 values for each other objects
                    spot = spot + object*5 

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
                            sentence.extend(["You","Push","Object"])
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
            if  obs[place] == 1 :

                #We update the area_obj
                self.map.update_area_obj(obs[0], obs[1],2,nb_agent)

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
                    sentence.extend(["I","Push","Object"])
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

        return sentence , push

    def landmarks_sentence(self, obs, sce_conf, nb_agent):

        sentence = []

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
            if  obs[place] == 1 :

                #We update the area_obj
                self.map.update_area_obj( obs[0], obs[1],3, nb_agent)

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
                     obs[place+1] < 0.2 and  obs[place+2] >= -0.2):
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

class ObservationParser(Parser):
    
    vocab = ['Located', 'Object', 'Landmark', 'North', 'South', 'East', 'West', 'Center', 'Not']
    def __init__(self, args):
        self.args = args

    #Check the position of the agent to see if it is in a corner
    def check_position(self, obs):
        if ( obs[0] >= 0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5) or
             obs[0] <= -0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5)):
            return True
        else:
            return False

    def objects_sentence(self, obs, sce_conf, not_sentence, position):

        sentence = []
        not_visible = []

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
                and  obs[place] == 0):
                    not_visible.append(place)

            # If visible                                      
            if  obs[place] == 1 :
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

        if len(not_visible) != 0:
            not_visible = random.choice(not_visible)
            sentence.extend(["Object","Not"])
            for word in position:
                sentence.append(word)

        return sentence

    def landmarks_sentence(self, obs, sce_conf, not_sentence, position):

        sentence = []
        not_visible = []

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

            """# If not visible and not sentence
            if ((not_sentence == 2 or not_sentence == 3) 
                and  obs[place] == 0):
                #if self.check_position(obs):
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Landmark","Not"])
                    for word in position:
                        sentence.append(word)"""

            if ((not_sentence == 2 or not_sentence == 3) 
                and  obs[place] == 0):
                    not_visible.append(place)

            # If visible
            if  obs[place] == 1 :
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
                     obs[place+1] < 0.2 and  obs[place+2] >= -0.2):
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

        if len(not_visible) != 0:
            not_visible = random.choice(not_visible)
            sentence.extend(["Landmark","Not"])
            for word in position:
                sentence.append(word)

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

        # Get the position of the agent
        sentence.extend(self.position_agent(obs))
        for i in range(1,len(sentence)):
            position.append(sentence[i])
        
        # Objects sentence
        sentence.extend(self.objects_sentence(obs, sce_conf, \
                        not_sentence, position))
        
        # Landmark sentence
        sentence.extend(self.landmarks_sentence(obs, sce_conf, \
                        not_sentence, position))

        return sentence

# Color parsers
class ObservationParserStratColor(ColorParser):
    
    vocab = ['Located', 'Object', 'Landmark', 'I', 'You', 'North', 'South', 'East', 'West', 'Center', 'Not', 'Push', 'Search', "Red", "Blue", "Yellow", "Green", "Black", "Purple"]

    def __init__(self, args, sce_conf, colors):
        self.args = args
        self.directions = []
        for nb_agent in range(sce_conf['nb_agents']):
            newAgent = []
            self.directions.append(newAgent)
        self.time = []
        for nb_agent in range(sce_conf['nb_agents']):
            self.time.append(0)

        self.map = ColorMapper(args, sce_conf)
        self.colors = colors

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
            print("Type: " + str(type))
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
        print("------------------------- NOT SENTENCE -------------------------")
        """print(self.map.area[nb_agent])
        print(self.map.area_obj)"""
        # Position of the agent
        position = []
        # Part of the sentence generated
        n_sent = []
        # Variable to check if you need to verify a whole area
        check = ""
        # Variable to check if the agent saw objects
        obj = 0
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
            objects = self.map.check_area(nb_agent, 0, 0, obj, self.colors)
            # Then we reset the area
            self.map.reset_areas(0, nb_agent)
        elif check == "South":
            objects = self.map.check_area(nb_agent, 0, 2, obj, self.colors)
            self.map.reset_areas(1, nb_agent)
        elif check == "West":
            objects = self.map.check_area(nb_agent, 1, 0, obj, self.colors)
            self.map.reset_areas(2, nb_agent)
        elif check == "East":
            objects = self.map.check_area(nb_agent, 1, 2, obj, self.colors)
            self.map.reset_areas(3, nb_agent)
        elif check == "Center":
            # Always reset the center
            self.map.reset_areas(4, nb_agent)
        # If we don't have to check a big area
        else :
            objects = self.map.find_missing(nb_agent, i, j, self.colors)

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
                    # 5 values for each other objects
                    spot = spot + object*6

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
                            sentence.extend(["You","Push",self.get_color(obs[spot+5]),"Object"])
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

        # Position of the objects
        # For each object 
        for object in range(int(sce_conf['nb_objects'])):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5 
            # 5 values for each other objects
            place = place + object*6 

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
                self.map.update_area_obj(obs[0], obs[1],2,obs[place+5],nb_agent)
                sentence.append(self.get_color(obs[place+5]))
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
                    sentence.extend(["I","Push",self.get_color(obs[place+5]),"Object"])
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

        return sentence , push

    def landmarks_sentence(self, obs, sce_conf, nb_agent):

        sentence = []

        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_objects'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5
            # 5 values for each objects 
            place = place + int(sce_conf['nb_objects'])*6 
            # 3 values for each landmark
            place = place + landmark*4

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
                self.map.update_area_obj( obs[0], obs[1],3, obs[place+3], nb_agent)
                sentence.append(self.get_color(obs[place+3]))
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
                     obs[place+1] < 0.2 and  obs[place+2] >= -0.2):
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
        print(self.map.area_object[0])
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

class ObservationParserColor(ColorParser):
    
    vocab = ['Located', 'Object', 'Landmark', 'North', 'South', 'East', 'West', 'Center', 'Not', "Red", "Blue", "Yellow", "Green", "Black", "Purple"]
    def __init__(self, args, colors):
        self.args = args
        self.colors = colors

    #Check the position of the agent to see if it is in a corner
    def check_position(self, obs):
        if ( obs[0] >= 0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5) or
             obs[0] <= -0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5)):
            return True
        else:
            return False

    def objects_sentence(self, obs, sce_conf, not_sentence, position):

        sentence = []
        not_visible = []
        obj = 0

        # Position of the objects
        # For each object 
        for object in range(int(sce_conf['nb_objects'])):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5 
            # 5 values for each other objects
            place = place + object*6 

            # If not visible and not sentence
            """if ((not_sentence == 1 or not_sentence == 3) 
                and  obs[place] == 0):
                #if self.check_position(obs) :
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Object","Not"])
                    for word in position:
                        sentence.append(word)"""
            if ((not_sentence == 1 or not_sentence == 3) 
                and  obs[place] == 0):
                    not_visible.append(obj)

            # If visible                                      
            if  obs[place] == 1 :
                sentence.append(self.get_color(obs[place+5]))
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

            obj += 1

        if len(not_visible) != 0:
            not_visible = random.choice(not_visible)
            """ --- Enregistrer les couleurs qu'ils voient pour cr??er les bonnes phrases ? --- """
            sentence.extend([self.get_color(self.colors[not_visible]),"Object","Not"])
            #sentence.extend(["Object","Not"])
            for word in position:
                sentence.append(word)

        return sentence

    def landmarks_sentence(self, obs, sce_conf, not_sentence, position):

        sentence = []
        not_visible = []
        obj = 0

        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_objects'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5
            # 5 values for each objects 
            place = place + int(sce_conf['nb_objects'])*6 
            # 3 values for each landmark
            place = place + landmark*4

            # If not visible and not sentence
            """if ((not_sentence == 2 or not_sentence == 3) 
                and  obs[place] == 0):
                #if self.check_position(obs):
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Landmark","Not"])
                    for word in position:
                        sentence.append(word)"""
            if ((not_sentence == 2 or not_sentence == 3) 
                and  obs[place] == 0):
                    not_visible.append(obj)

            # If visible
            if  obs[place] == 1 :
                sentence.append(self.get_color(obs[place+3]))
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
                     obs[place+1] < 0.2 and  obs[place+2] >= -0.2):
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

        if len(not_visible) != 0:
            not_visible = random.choice(not_visible)
            """ --- Enregistrer les couleurs qu'ils voient pour cr??er les bonnes phrases ? --- """
            sentence.extend([self.get_color(self.colors[not_visible]),"Landmark","Not"])
            for word in position:
                sentence.append(word)

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

        # Get the position of the agent
        sentence.extend(self.position_agent(obs))
        for i in range(1,len(sentence)):
            position.append(sentence[i])
        
        # Objects sentence
        sentence.extend(self.objects_sentence(obs, sce_conf, \
                        not_sentence, position))
        
        # Landmark sentence
        sentence.extend(self.landmarks_sentence(obs, sce_conf, \
                        not_sentence, position))

        return sentence

# Color and Shape parsers
class ObservationParserStratColorShape(ColorShapeParser):
    
    vocab = ['Located', 'Object', 'Landmark', 'I', 'You', 'North', 'South', 'East', 'West', 'Center', 'Not', 'Push', 'Search', "Red", "Blue", "Yellow", "Green", "Black", "Purple", "Circle", "Square", "Triangle"]

    def __init__(self, args, sce_conf):
        self.args = args
        self.directions = []
        for nb_agent in range(sce_conf['nb_agents']):
            newAgent = []
            self.directions.append(newAgent)
        self.time = []
        for nb_agent in range(sce_conf['nb_agents']):
            self.time.append(0)

        self.map = Mapper(args, sce_conf)

    def not_sentence(self, i, j, nb_agent):
        """print("------------------------- NOT SENTENCE -------------------------")
        print(self.map.area[nb_agent])
        print(self.map.area_obj)"""
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
                    # 5 values for each other objects
                    spot = spot + object*7

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
                            sentence.extend(["You","Push",self.get_color(obs[spot+5]),self.get_shape(obs[spot+6]),"Object"])
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

        # Position of the objects
        # For each object 
        for object in range(int(sce_conf['nb_objects'])):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5 
            # 5 values for each other objects
            place = place + object*7 

            # If visible                                      
            if  obs[place] == 1 :

                #We update the area_obj
                self.map.update_area_obj(obs[0], obs[1],2,nb_agent)
                sentence.append(self.get_color(obs[place+5]))
                sentence.append(self.get_shape(obs[place+6]))
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
                    sentence.extend(["I","Push",self.get_color(obs[place+5]),self.get_shape(obs[place+6]),"Object"])
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

        return sentence , push

    def landmarks_sentence(self, obs, sce_conf, nb_agent):

        sentence = []

        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_objects'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5
            # 5 values for each objects 
            place = place + int(sce_conf['nb_objects'])*7 
            # 3 values for each landmark
            place = place + landmark*5

            # If visible
            if  obs[place] == 1 :

                #We update the area_obj
                self.map.update_area_obj( obs[0], obs[1],3, nb_agent)
                sentence.append(self.get_color(obs[place+3]))
                sentence.append(self.get_shape(obs[place+4]))
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
                     obs[place+1] < 0.2 and  obs[place+2] >= -0.2):
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

class ObservationParserColorShape(ColorShapeParser):
    
    vocab = ['Located', 'Object', 'Landmark', 'North', 'South', 'East', 'West', 'Center', 'Not', "Red", "Blue", "Yellow", "Green", "Black", "Purple", "Circle", "Square", "Triangle"]
    def __init__(self, args, colors, shapes):
        self.args = args
        self.colors = colors
        self.shapes = shapes

    #Check the position of the agent to see if it is in a corner
    def check_position(self, obs):
        if ( obs[0] >= 0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5) or
             obs[0] <= -0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5)):
            return True
        else:
            return False

    def objects_sentence(self, obs, sce_conf, not_sentence, position):

        sentence = []
        not_visible = []
        obj = 0

        # Position of the objects
        # For each object 
        for object in range(int(sce_conf['nb_objects'])):
        # Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5 
            # 5 values for each other objects
            place = place + object*7 

            # If not visible and not sentence
            """if ((not_sentence == 1 or not_sentence == 3) 
                and  obs[place] == 0):
                #if self.check_position(obs) :
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Object","Not"])
                    for word in position:
                        sentence.append(word)"""
            if ((not_sentence == 1 or not_sentence == 3) 
                and  obs[place] == 0):
                    not_visible.append(obj)

            # If visible                                      
            if  obs[place] == 1 :
                sentence.append(self.get_color(obs[place+5]))
                sentence.append(self.get_shape(obs[place+6]))
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
            
            obj += 1

        if len(not_visible) != 0:
            not_visible = random.choice(not_visible)
            sentence.extend([self.get_color(self.colors[not_visible]),self.get_shape(self.shapes[not_visible]),"Object","Not"])
            for word in position:
                sentence.append(word)

        return sentence

    def landmarks_sentence(self, obs, sce_conf, not_sentence, position):

        sentence = []
        not_visible = []
        obj = 0

        # Position of the Landmarks
        # For each Landmark 
        for landmark in range(int(sce_conf['nb_objects'])):
        #Calculate the place in the array
            place = 4 # 4 values of the self agent
            # 5 values for each agents (not self)
            place = place + (int(sce_conf['nb_agents'])-1)*5
            # 5 values for each objects 
            place = place + int(sce_conf['nb_objects'])*7 
            # 3 values for each landmark
            place = place + landmark*5

            # If not visible and not sentence
            """if ((not_sentence == 2 or not_sentence == 3) 
                and  obs[place] == 0):
                #if self.check_position(obs):
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Landmark","Not"])
                    for word in position:
                        sentence.append(word)"""
            if ((not_sentence == 2 or not_sentence == 3) 
                and  obs[place] == 0):
                    not_visible.append(obj)

            # If visible
            if  obs[place] == 1 :
                sentence.append(self.get_color(obs[place+3]))
                sentence.append(self.get_shape(obs[place+4]))
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
                     obs[place+1] < 0.2 and  obs[place+2] >= -0.2):
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

        if len(not_visible) != 0:
            not_visible = random.choice(not_visible)
            sentence.extend([self.get_color(self.colors[not_visible]),self.get_shape(self.shapes[not_visible]),"Landmark","Not"])
            for word in position:
                sentence.append(word)

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

        # Get the position of the agent
        sentence.extend(self.position_agent(obs))
        for i in range(1,len(sentence)):
            position.append(sentence[i])
        
        # Objects sentence
        sentence.extend(self.objects_sentence(obs, sce_conf, \
                        not_sentence, position))
        
        # Landmark sentence
        sentence.extend(self.landmarks_sentence(obs, sce_conf, \
                        not_sentence, position))

        return sentence