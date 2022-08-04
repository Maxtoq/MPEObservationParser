import random
from math import sqrt

from abc import ABC, abstractmethod

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

    """@abstractmethod
    def objects_sentence(self, obs, sce_conf):
        
        Returns a sentence generated based on the objects see or not by the agent
        
        raise NotImplementedError"""

    @abstractmethod
    def landmarks_sentence(self, obs, sce_conf):
        """
        Returns a sentence generated based on the landmarks see or not by the agent
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, sce_conf, colors, shapes):
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

