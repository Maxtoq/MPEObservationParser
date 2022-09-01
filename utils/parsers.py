from math import sqrt
import numpy as np

from abc import ABC, abstractmethod

# Mother classes
class Parser(ABC):
    """ Abstract Parser """
    @abstractmethod
    def parse_obs(self, obs):
        """
        Returns a sentence generated based on the actions of the agent
        """
        raise NotImplementedError

    # Generate a sentence on the position of the agent
    def position_agent(self, obs):
        '''
        Generate a sentence on the position of the agent

        Input: 
            obs: list(float) observation of the agent 

        Output: 
            list(str) The sentence generated
        '''
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
    def reset(self, obj_colors, obj_shapes, land_colors, land_shapes):
        """
        Returns a sentence generated based on the landmarks see or not by the agent
        """
        raise NotImplementedError

class ColorParser(Parser):
    """ Abstract Parser """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    # Get the color based on the number
    def get_color(self, color):
        # Red
        if color == 1:
            color = "Red"
        # Blue
        elif color == 2:
            color = "Bleu"
        # Green
        elif color == 3:
            color = "Green"

        return color

    # Get the color based on its array
    def array_to_color(self, array):
        # Get the color based on the array
        idx = np.where(array == 1)[0]
        color = None
        # Red
        if idx == 0:
            color = "Red"
        # Blue
        elif idx == 1:
            color = "Blue"
        # Green
        elif idx == 2:
            color = "Green"

        return color

    # Get the color number based on its array
    def array_to_num(self, array):
        # Get the color based on the array
        idx = np.where(array == 1)[0]
        color = 0
        # Red
        if idx == 0:
            color = 1
        # Blue
        elif idx == 1:
            color = 2
        # Green
        elif idx == 2:
            color = 3

        return color
        
class ColorShapeParser(ColorParser):
    """ Abstract Parser """
    # Get the shape based on the number
    def get_shape(self, shape):
        #Black
        if shape == 1:
            shape = "Circle"
        # Red
        elif shape == 2:
            shape = "Square"
        # Blue
        elif shape == 3:
            shape = "Triangle"

        return shape