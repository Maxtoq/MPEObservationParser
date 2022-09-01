import keyboard

class Render_option:

    def __init__(self) :
        """
        Set the range to False for each agent
        if == True : show the range on the environment
        """
        self.range_a1 = False
        self.range_a2 = False

    # If a key is pressed, will toggle the value of the range     
    def modify_option(self):
        """
        Toggle the value of range
        """
        # Show / Hide the range of the agents 
        # Agent 1
        if keyboard.is_pressed('1'):
            self.range_a1 = not self.range_a1
        # Agent 2
        if keyboard.is_pressed('3'):
            self.range_a2 = not self.range_a2

        return self.range_a1, self.range_a2
