import random
from math import sqrt

from abc import ABC, abstractmethod
from re import S


class Parser(ABC):
    """ Abstract Parser """
    @abstractmethod
    def parse_obs(self, obs, sce_conf):
        """
        Returns a sentence generated based on the actions of the agent
        """
        raise NotImplementedError

class ObservationParserStrat(Parser):
    
    def __init__(self, args, sce_conf):
        self.args = args
        self.directions = []
        for nb_agent in range(sce_conf['nb_agents']):
            newAgent = []
            self.directions.append(newAgent)
        self.time = []
        for nb_agent in range(sce_conf['nb_agents']):
            self.time.append(0)

        # Initialization of the world map
        # To track the agent
        self.world = []
        for nb_agent in range(sce_conf['nb_agents']):
            newAgent = []
            for line in range(6):
                newLine = []
                for col in range(6):
                    newLine.append(0)
                newAgent.append(newLine)
            self.world.append(newAgent)
        

        # Initialization of the area map
        # Each 0 is an area (North, South, West, East)
        # That has not been fully discovered
        self.area = []
        for nb_agent in range(sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    newLine.append(0)
                newAgent.append(newLine)
            self.area.append(newAgent)

        # Initialization of the area map with objects
        # Each 0 is the number of objects found in the area
        self.area_obj = []
        for nb_agent in range(sce_conf['nb_agents']):
            newAgent = []
            for line in range(3):
                newLine = []
                for col in range(3):
                    newLine.append(0)
                newAgent.append(newLine)
            self.area_obj.append(newAgent)

    def update_world_section(self, nb_agent, num_array, posX, corner):
        # 0 means not discovered
        # 1 means discovered
        # 2 means discovered in corners

        if corner == True:
            # In a corner so = 2 (for the two possible areas)
            if posX <= -0.66:
                self.world[nb_agent][num_array][0] = 2
            if posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][num_array][1] = 2
            if posX >= -0.33 and posX <= 0:
                self.world[nb_agent][num_array][2] = 1
            if posX >= 0 and posX <= 0.33:
                self.world[nb_agent][num_array][3] = 1
            if posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][num_array][4] = 2
            if posX >= 0.66:
                self.world[nb_agent][num_array][5] = 2
        else:
            if posX <= -0.66:
                self.world[nb_agent][num_array][0] = 1
            if posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][num_array][1] = 1
            if posX >= -0.33 and posX <= 0:
                self.world[nb_agent][num_array][2] = 1
            if posX >= 0 and posX <= 0.33:
                self.world[nb_agent][num_array][3] = 1
            if posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][num_array][4] = 1
            if posX >= 0.66:
                self.world[nb_agent][num_array][5] = 1

    def update_world(self, posX, posY, nb_agent):
        #Check the position of the agent
        # To update the value of the world
        
        # North
        # Can be in a corner
        if posY >= 0.66 :
            self.update_world_section(nb_agent,0,posX,corner = True)
        if posY >= 0.33 and posY <= 0.66 :
            self.update_world_section(nb_agent,1,posX,corner = True)

        # Center
        if posY >= 0 and posY <= 0.33 :
            self.update_world_section(nb_agent,2,posX,corner = False)
        if posY >= -0.33 and posY <= 0 :
            self.update_world_section(nb_agent,3,posX,corner = False)

        # South 
        # Can be in a corner
        if posY >= -0.66 and posY <= -0.33 :
            self.update_world_section(nb_agent,4,posX,corner = True)
        if posY <= -0.66:
            self.update_world_section(nb_agent,5,posX,corner = True)

        # To see what the agent saw
        for l in range(6) :   
            print(self.world[nb_agent][l])


        """# North
        if posY >= 0.66 :
            if posX <= -0.66:
                # In a corner so = 2 (for the two possible areas)
                # North or West
                self.world[nb_agent][0][0] = 2
            if posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][0][1] = 2
            if posX >= -0.33 and posX <= 0:
                self.world[nb_agent][0][2] = 1
            if posX >= 0 and posX <= 0.33:
                self.world[nb_agent][0][3] = 1
            if posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][0][4] = 2
            if posX >= 0.66:
                self.world[nb_agent][0][5] = 2
        if posY >= 0.33 and posY <= 0.66 :
            if posX <= -0.66:
                self.world[nb_agent][1][0] = 2
            if posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][1][1] = 2
            if posX >= -0.33 and posX <= 0:
                self.world[nb_agent][1][2] = 1
            if posX >= 0 and posX <= 0.33:
                self.world[nb_agent][1][3] = 1
            if posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][1][4] = 2
            if posX >= 0.66 :
                self.world[nb_agent][1][5] = 2

        # Center
        if posY >= 0 and posY <= 0.33 :
            if posX <= -0.66:
                self.world[nb_agent][2][0] = 1
            if posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][2][1] = 1
            if posX >= -0.33 and posX <= 0:
                self.world[nb_agent][2][2] = 1
            if posX >= 0 and posX <= 0.33:
                self.world[nb_agent][2][3] = 1
            if posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][2][4] = 1
            if posX >= 0.66:
                self.world[nb_agent][2][5] = 1
        if posY >= -0.33 and posY <= 0 :
            if posX <= -0.66:
                self.world[nb_agent][3][0] = 1
            if posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][3][1] = 1
            if posX >= -0.33 and posX <= 0:
                self.world[nb_agent][3][2] = 1
            if posX >= 0 and posX <= 0.33:
                self.world[nb_agent][3][3] = 1
            if posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][3][4] = 1
            if posX >= 0.66 :
                self.world[nb_agent][3][5] = 1

        # South
        if posY >= -0.66 and posY <= -0.33 :
            if posX <= -0.66:
                self.world[nb_agent][4][0] = 2
            if posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][4][1] = 2
            if posX >= -0.33 and posX <= 0:
                self.world[nb_agent][4][2] = 1
            if posX >= 0 and posX <= 0.33:
                self.world[nb_agent][4][3] = 1
            if posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][4][4] = 2
            if posX >= 0.66:
                self.world[nb_agent][4][5] = 2
        if posY <= -0.66:
            if posX <= -0.66:
                self.world[nb_agent][5][0] = 2
            if posX >= -0.66 and posX <= -0.33:
                self.world[nb_agent][5][1] = 2
            if posX >= -0.33 and posX <= 0:
                self.world[nb_agent][5][2] = 1
            if posX >= 0 and posX <= 0.33:
                self.world[nb_agent][5][3] = 1
            if posX >= 0.33 and posX <= 0.66:
                self.world[nb_agent][5][4] = 2
            if posX >= 0.66 :
                self.world[nb_agent][5][5] = 2

        # To see what the agent saw
        for l in range(6) :   
            print(self.world[nb_agent][l])"""

    def count_discovered(self, nb_agent, world_array_posx, world_array_posy):
        max_count = 0
        count = 0

        # Count the number of cell discovered
        for i in range(world_array_posx[0], world_array_posx[1]):
            for j in range(world_array_posy[0], world_array_posy[1]):
                max_count += 1
                # if == 1 or == 2, the agent saw it
                if self.world[nb_agent][i][j] >= 1 :
                    count += 1
                else:
                    break
        
        # Returns True or False depending on 
        # If the agent saw it entirely
        if count == max_count:
            return True
        else:
            return False

    def count_discovered_corner(self, nb_agent, world_array_posx, world_array_posy):
        max_count = 0
        count = 0

        # Count the number of cell discovered (corner)
        for i in range(world_array_posx[0], world_array_posx[1]):
            for j in range(world_array_posy[0], world_array_posy[1]):
                max_count += 1
                # if == 2, the agent saw the corner
                if self.world[nb_agent][i][j] > 1 :
                    count += 1
                else:
                    break
        
        # Returns True or False depending on 
        # If the agent saw it entirely
        if count == max_count:
            return True
        else:
            return False

    def update_area(self, nb_agent):
        # Check the world to see if some area were fully discovered
        print(self.area[nb_agent])
        # If North is not fully discovered
        if (self.area[nb_agent][0][0] < 1 or self.area[nb_agent][0][1] < 1 or 
        self.area[nb_agent][0][2] < 1):
            # North West 
            if self.area[nb_agent][0][0] < 2 :
                if self.count_discovered_corner(nb_agent,[0,2],[0,2]):
                        self.area[nb_agent][0][0] = 2
                        # Generate a not sentence
                        return self.not_sentence(0,0, nb_agent)
            # North Center
            if self.area[nb_agent][0][1] != 1:
                if self.count_discovered(nb_agent,[0,2],[2,4]):
                    if self.area[nb_agent][0][1] == 0:
                        self.area[nb_agent][0][1] = 1
            # North East
            if self.area[nb_agent][0][2] < 2:
                if self.count_discovered_corner(nb_agent,[0,2],[4,6]):
                        self.area[nb_agent][0][2] = 2
                        # Generate a not sentence
                        return self.not_sentence(0,2, nb_agent)
        else:
            return self.not_sentence(0,1, nb_agent)

        # If Center not fully discovered
        if (self.area[nb_agent][1][0] != 1 or self.area[nb_agent][1][1] != 1 or 
        self.area[nb_agent][1][2] != 1):
            # Center West 
            if self.area[nb_agent][1][0] != 1:
                if self.count_discovered(nb_agent,[2,4],[0,2]):
                    if self.area[nb_agent][1][0] == 0:
                        self.area[nb_agent][1][0] = 1
            # Center Center
            if self.area[nb_agent][1][1] != 1:
                if self.count_discovered(nb_agent,[2,4],[2,4]):
                    if self.area[nb_agent][1][1] == 0:
                        self.area[nb_agent][1][1] = 1
                    return self.not_sentence(1,1, nb_agent)
            # Center East
            if self.area[nb_agent][1][2] != 1:
                if self.count_discovered(nb_agent,[2,4],[4,6]):
                    if self.area[nb_agent][1][2] == 0:
                        self.area[nb_agent][1][2] = 1

        # If South is not fully discovered
        if (self.area[nb_agent][2][0] < 1 or self.area[nb_agent][2][1] < 1 or 
        self.area[nb_agent][2][2] < 1):
            # South West 
            if self.area[nb_agent][2][0] < 2:
                if self.count_discovered_corner(nb_agent,[4,6],[0,2]):
                        self.area[nb_agent][2][0] = 2
                        return self.not_sentence(2,0, nb_agent)
            # South Center
            if self.area[nb_agent][2][1] != 1:
                if self.count_discovered(nb_agent,[4,6],[2,4]):
                    if self.area[nb_agent][2][1] == 0:
                        self.area[nb_agent][2][1] = 1
            # South East
            if self.area[nb_agent][2][2] < 2:
                if self.count_discovered_corner(nb_agent,[4,6],[4,6]):
                        self.area[nb_agent][2][2] = 2
                        return self.not_sentence(2,2, nb_agent)
        else:
            return self.not_sentence(2,1, nb_agent)

        # West and East
        # If the 3 areas were discovered
        if (self.area[nb_agent][0][0] >= 1 and self.area[nb_agent][1][0] >= 1 and 
        self.area[nb_agent][2][0] >= 1):
            # Generate the not_sentence
            return self.not_sentence(1,0, nb_agent)
        if (self.area[nb_agent][0][2] >= 1 and self.area[nb_agent][1][2] >= 1 and 
        self.area[nb_agent][2][2] >= 1):
            return self.not_sentence(1,2, nb_agent)

    def reset_area(self, nb_agent, direction, area_pos):
        # If direction is left to right (North or South)
        if direction == 0:
            for i in range(3):
                # Reset the area by doing -1
                self.area[nb_agent][area_pos][i] -= 1
                # If the agent saw an object (and in a corner)
                if self.area_obj[nb_agent][area_pos][i] >= 4:
                    # Devide it by 2
                    self.area_obj[nb_agent][area_pos][i] = \
                        self.area_obj[nb_agent][area_pos][i]//2
                else:
                    # Or = 0
                    self.area_obj[nb_agent][area_pos][i] = 0

        # If direction is up to bottom (West or East)
        elif direction == 1:
            for i in range(3):
                self.area[nb_agent][i][area_pos] -= 1
                if self.area_obj[nb_agent][i][area_pos] >= 4:
                    self.area_obj[nb_agent][i][area_pos] = \
                        self.area_obj[nb_agent][i][area_pos]//2
                else:
                    self.area_obj[nb_agent][i][area_pos] = 0

    def reset_world(self, nb_agent, world_array_posx, world_array_posy):
        # Reset the world map depending on the array position
        for i in range(world_array_posx[0],world_array_posx[1]) :
                for j in range(world_array_posy[0],world_array_posy[1]):
                    self.world[nb_agent][i][j] -= 1 

    def reset_areas(self, area_nb, nb_agent):

        # If North
        if area_nb == 0:
            # Reset the area and the world map
            # Send area values to other programs
            self.reset_area(nb_agent,0,0)
            self.reset_world(nb_agent,[0,2],[0,6])
        # If South
        if area_nb == 1:
            self.reset_area(nb_agent,0,2)
            self.reset_world(nb_agent,[4,6],[0,6])
        # if West
        if area_nb == 2:
            self.reset_area(nb_agent,1,0)
            self.reset_world(nb_agent,[0,6],[0,2])
        # If East
        if area_nb == 3:
            self.reset_area(nb_agent,1,2)
            self.reset_world(nb_agent,[0,6],[4,6])
        
        # If Center
        if area_nb == 4:
            # Reset the area (1,1)
            self.area[nb_agent][1][1] -= 1
            if self.area_obj[nb_agent][1][1] >= 4:
                self.area_obj[nb_agent][1][1] = \
                    self.area_obj[nb_agent][1][1]//2
            else:
                self.area_obj[nb_agent][1][1] = 0
            # Reset the world
            self.reset_world(nb_agent,[2,4],[2,4])

        """# If the area was North
        if area_nb == 0:
            for i in range(3):
                # Set the area back to 0
                self.area[nb_agent][0][i] -= 1
                # If the agent saw an object (and in a corner)
                if self.area_obj[nb_agent][0][i] >= 4:
                    # Devide it by 2
                    self.area_obj[nb_agent][0][i] = self.area_obj[nb_agent][0][i]//2
                else:
                    # Or = 0
                    self.area_obj[nb_agent][0][i] = 0
            # Update the world (1-1=0 or, in a corner, 2-1 = 1)
            for i in range(2) :
                for j in range(6):
                    self.world[nb_agent][i][j] -= 1 

        # If the area was South
        if area_nb == 1:
            for i in range(3):
                self.area[nb_agent][2][i] -= 1
            if self.area_obj[nb_agent][2][i] >= 4:
                self.area_obj[nb_agent][2][i] = self.area_obj[nb_agent][2][i]//2
            else:
                self.area_obj[nb_agent][2][i] = 0
            for i in range(4,6) :
                for j in range(6):
                    self.world[nb_agent][i][j] -= 1 

        # If the area was West
        if area_nb == 2:
            for i in range(3):
                self.area[nb_agent][i][0] -= 1
                if self.area_obj[nb_agent][i][0] >= 4:
                    self.area_obj[nb_agent][i][0] = self.area_obj[nb_agent][i][0]//2
                else:
                    self.area_obj[nb_agent][i][0] = 0
            for i in range(2) :
                for j in range(6):
                    self.world[nb_agent][j][i] -= 1 

        # If the area was East
        if area_nb == 3:
            for i in range(3):
                self.area[nb_agent][i][2] -= 1
                if self.area_obj[nb_agent][i][2] >=4:
                    self.area_obj[nb_agent][i][2] = self.area_obj[nb_agent][i][2]//2
                else:
                    self.area_obj[nb_agent][i][2] = 0
            for i in range(4,6) :
                for j in range(6):
                    self.world[nb_agent][j][i] -= 1 

        # If the area was Center
        if area_nb == 4:
            self.area[nb_agent][1][1] -= 1
            if self.area_obj[nb_agent][1][1] >= 4:
                self.area_obj[nb_agent][1][1] = self.area_obj[nb_agent][1][1]//2
            else:
                self.area_obj[nb_agent][1][1] = 0
            for i in range(2,4):
                for j in range(2,4):
                    self.world[nb_agent][i][j] -= 1"""

    def check_area(self, nb_agent, direction, area_pos, obj):
        
        # If North or South
        if direction == 0:
            for x in range(3) :
                    # obj_i represente the object found in the area
                    obj_i = self.area_obj[nb_agent][area_pos][x]
                    # If 2 or 4 then obj = 2
                    if ((obj_i == 2 or obj_i == 4) and 
                    (obj == 2 or obj == 4 or obj == 0)):
                        obj = 2
                    # if 3 or 6 then obj = 3
                    elif ((obj_i == 3 or obj_i == 6) and 
                    (obj == 3 or obj == 6 or obj == 0)):
                        obj = 3
                    elif obj_i != 0 :
                        # Else, it means that two different objects
                        # Are in the same area, obj = 4
                        obj = 4
        # If West or East
        elif direction == 1:
            for x in range(3) :
                    obj_i = self.area_obj[nb_agent][x][area_pos]
                    if ((obj_i == 2 or obj_i == 4) and 
                    (obj == 2 or obj == 4 or obj == 0)):
                        obj = 2
                    elif ((obj_i == 3 or obj_i == 6) and 
                    (obj == 3 or obj == 6 or obj == 0)):
                        obj = 3
                    elif obj_i != 0 :
                        obj = 4

        return obj

    def not_sentence(self, i, j, nb_agent):
        print("------------------------- NOT SENTENCE -------------------------")
        print(self.area[nb_agent])
        print(self.area_obj)
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
        obj = self.area_obj[nb_agent][i][j]
        if obj >=4 and obj != 5:
            obj = obj//2
            
        # If the agent discovered a whole area
        # We have to check all 3 areas
        if check == "North":
            # Update the object value depending on the objects in the area
            obj = self.check_area(nb_agent, 0, 0, obj)
            # Then we reset the area
            self.reset_areas(0, nb_agent)
        elif check == "South":
            obj = self.check_area(nb_agent, 0, 2, obj)
            self.reset_areas(1, nb_agent)
        elif check == "West":
            obj = self.check_area(nb_agent, 1, 0, obj)
            self.reset_areas(2, nb_agent)
        elif check == "East":
            obj = self.check_area(nb_agent, 1, 2, obj)
            self.reset_areas(3, nb_agent)
        elif check == "Center":
            # Always reset the center
            self.reset_areas(4, nb_agent)

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
            
    def update_area_obj(self, agent_x, agent_y, object_nb, nb_agent):
        # object_nb : 2 if object
        #       3 if landmark
        #       4 if both
        """print("OBJECT FOUND: " + str(object_nb))
        print(self.area)
        print(self.area_obj[nb_agent])"""
        
        # If an object is discovered, we modify the area_ibj array
        
        # North
        if agent_y >= 0.33:
            if agent_x >= 0.33:
                # If the object is different than
                # An object seen in the same area
                if (self.area_obj[nb_agent][0][2] != 0 and 
                self.area_obj[nb_agent][0][2] != object_nb and 
                self.area_obj[nb_agent][0][2] != object_nb*2):
                    self.area_obj[nb_agent][0][2] = 5
                # Else, we are in a corner
                # So we multiply the object num by 2
                else :
                    self.area_obj[nb_agent][0][2] = object_nb*2
            elif agent_x <= -0.33:
                if (self.area_obj[nb_agent][0][0] != 0 and 
                self.area_obj[nb_agent][0][0] != object_nb and 
                self.area_obj[nb_agent][0][0] != object_nb*2):
                    self.area_obj[nb_agent][0][0] = 5
                else :
                    self.area_obj[nb_agent][0][0] = object_nb*2
            else :
                if (self.area_obj[nb_agent][0][1] != 0 and 
                self.area_obj[nb_agent][0][1] != object_nb and 
                self.area_obj[nb_agent][0][1] != object_nb*2):
                    self.area_obj[nb_agent][0][1] = 5
                else :
                    self.area_obj[nb_agent][0][1] = object_nb
        # South
        elif agent_y <= -0.33:
            if agent_x >= 0.33:
                if (self.area_obj[nb_agent][2][2] != 0 and 
                self.area_obj[nb_agent][2][2] != object_nb and 
                self.area_obj[nb_agent][2][2] != object_nb*2):
                    self.area_obj[nb_agent][2][2] = 5
                else :
                    self.area_obj[nb_agent][2][2] = object_nb*2
            elif agent_x <= -0.33:
                if (self.area_obj[nb_agent][2][0] != 0 and 
                self.area_obj[nb_agent][2][0] != object_nb and 
                self.area_obj[nb_agent][2][0] != object_nb*2):
                    self.area_obj[nb_agent][2][0] = 5
                else :
                    self.area_obj[nb_agent][2][0] = object_nb*2
            else :
                if (self.area_obj[nb_agent][2][1] != 0 and 
                self.area_obj[nb_agent][2][1] != object_nb and 
                self.area_obj[nb_agent][2][1] != object_nb*2):
                    self.area_obj[nb_agent][2][1] = 5
                else :
                    self.area_obj[nb_agent][2][1] = object_nb
        
        # Center
        else:
            if agent_x >= 0.33:
                if (self.area_obj[nb_agent][1][2] != 0 and 
                self.area_obj[nb_agent][1][2] != object_nb and 
                self.area_obj[nb_agent][1][2] != object_nb*2):
                    self.area_obj[nb_agent][1][2] = 5
                else :
                    self.area_obj[nb_agent][1][2] = object_nb
            elif agent_x <= -0.33:
                if (self.area_obj[nb_agent][1][0] != 0 and 
                self.area_obj[nb_agent][1][0] != object_nb and 
                self.area_obj[nb_agent][1][0] != object_nb*2):
                    self.area_obj[nb_agent][1][0] = 5
                else :
                    self.area_obj[nb_agent][1][0] = object_nb
            else :
                if (self.area_obj[nb_agent][1][1] != 0 and 
                self.area_obj[nb_agent][1][1] != object_nb and 
                self.area_obj[nb_agent][1][1] != object_nb*2):
                    self.area_obj[nb_agent][1][1] = 5
                else :
                    self.area_obj[nb_agent][1][1] = object_nb

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

    def position_agent(self, obs):
        sentence = []
        # Position of the agent (at all time)
        sentence.append("Located")
        
        # North / South
        if  obs[1] >= 0.32:
            sentence.append("North")
        if  obs[1] < -0.32:
            sentence.append("South")
        
        # West / East
        if  obs[0] >= 0.32:
            sentence.append("East")
        if  obs[0] < -0.32:
            sentence.append("West")
        
        # Center
        elif len(sentence) == 1:
            sentence.append("Center")

        return sentence

    def parse_obs(self, obs, sce_conf, nb_agent):
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []
        # If the action of pushing happens
        push = False
        
        # Direction of the agent
        direction = []

        # Get the position of the agent
        sentence.extend(self.position_agent(obs))
        for i in range(1,len(sentence)):
            position.append(sentence[i])
        
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
                    if  obs[place+4] > 0.5:
                        sentence.append("North")
                        search = True
                    if  obs[place+4] < -0.5:
                        sentence.append("South")
                        search = True
                    if  obs[place+3] > 0.5:
                        sentence.append("East")
                        search = True
                    if  obs[place+3] < -0.5:
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
            if  obs[place] == 1 :

                #We update the area_obj
                #self.update_area_obj( obs[0], obs[1],2)
                self.update_area_obj( obs[0], obs[1],2,nb_agent)

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

                # Calculate the distance of the center of the object from the agent
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
                #self.update_area_obj( obs[0], obs[1],3)
                self.update_area_obj( obs[0], obs[1],3, nb_agent)

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


        # Update the world variable to see what the agent
        # Has discovered (to generate not sentences)
        self.update_world( obs[0], obs[1], nb_agent)
        temp = self.update_area(nb_agent)
        if temp != None:
            sentence.extend(temp)

        return sentence


class ObservationParser(Parser):
    
    def __init__(self, args):
        self.args = args

    #Check the position of the agent to see if it is in a corner
    def check_position(self, obs):
        if ( obs[0] >= 0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5) or
             obs[0] <= -0.5 and ( obs[1] >=0.5 or  obs[1] <= -0.5)):
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


        # Position of the agent (at all time)
        sentence.append("Located")
        
        # North / South
        if  obs[1] >= 0.32:
            sentence.append("North")
            position.append("North")
        if  obs[1] < -0.32:
            sentence.append("South")
            position.append("South")
        
        # West / East
        if  obs[0] >= 0.32:
            sentence.append("East")
            position.append("East")
        if  obs[0] < -0.32:
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
                and  obs[place] == 0):
                #if self.check_position(obs) :
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Object","Not"])
                    for word in position:
                        sentence.append(word)

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
                and  obs[place] == 0):
                #if self.check_position(obs):
                    # [1] and [2] are the positions of the agent
                    sentence.extend(["Landmark","Not"])
                    for word in position:
                        sentence.append(word)

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

        return sentence