from random import randint

class Apple():
    def __init__(self, x=5, y=5, step=50):
        self.step = step
        self.x = x * self.step
        self.y = y * self.step

    def draw(self, surface, image):
        """Show apple in pygame app"""
        surface.blit(image, (self.x, self.y))

class Game():
    def __init__(self, window_width=500, window_height=500, apple=Apple()):
        self.window_width = window_width
        self.window_height = window_height
        self.apple = apple

    def place_apple(self):
        """Ramdom x,y coordinate for apple"""
        # place apple
        self.apple = Apple(randint(0, self.window_width / self.apple.step - 1),
                           randint(0, self.window_height / self.apple.step - 1))

    def apple_location(self):
        return [self.apple.x, self.apple.y]