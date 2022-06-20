import keyboard

class Render_option:

    def __init__(self) :
        self.range_a1 = False
        self.range_a2 = False
        
    def modify_option(self):
        # Show / Hide the range of the agents 
        if keyboard.is_pressed('1'):
            self.range_a1 = not self.range_a1
            print(self.range_a1)
        if keyboard.is_pressed('2'):
            self.range_a2 = not self.range_a2

        return self.range_a1, self.range_a2
