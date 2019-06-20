from pygame.locals import *
from random import randint
import pygame
import time


class Apple():

    def __init__(self, x=5, y=5, step=50):
        self.step = step
        self.x = x * self.step
        self.y = y * self.step

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))


class Snake():

    def __init__(self, step=50, direction=0, length=5,
                 update_count_max=2, update_count=0):
        self.step = step
        self.direction = direction
        self.length = length
        self.update_count_max = update_count_max
        self.update_count = update_count

        # initial positions, no collision.
        self.x = [(i + 1) * self.step for i in range(self.length - 1, -1, -1)]
        self.y = [1 * self.step for i in range(self.length - 1, -1, -1)]

    def update(self):
        #         print(list(zip(self.x,self.y)))
        #         print(self.update_count)

        self.update_count = self.update_count + 1
        if self.update_count > self.update_count_max:

            # update previous positions
            for i in range(self.length - 1, 0, -1):
                self.x[i] = self.x[i - 1]
                self.y[i] = self.y[i - 1]

            # update position of head of snake
            if self.direction == 0:
                self.x[0] = self.x[0] + self.step
            if self.direction == 1:
                self.x[0] = self.x[0] - self.step
            if self.direction == 2:
                self.y[0] = self.y[0] - self.step
            if self.direction == 3:
                self.y[0] = self.y[0] + self.step

            self.update_count = 0

    def moveRight(self):
        self.direction = 0

    def moveLeft(self):
        self.direction = 1

    def moveUp(self):
        self.direction = 2

    def moveDown(self):
        self.direction = 3

    def draw(self, surface, image):
        for i in range(0, self.length):
            surface.blit(image, (self.x[i], self.y[i]))


class Game():

    def __init__(self, window_width=500, window_height=500, player=Snake(), apple=Apple()):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.window_width = window_width
        self.window_height = window_height
        self.player = player
        self.apple = apple

    def is_collision(self, x1, y1, x2, y2):
        if ((x1 == x2) and (y1 == y2)):
            return True
        return False

    def out_of_bounds(self, x1, y1):
        if ((x1 < 0) or (y1 < 0) or
                (x1 > self.window_width-self.player.step) or (y1 > self.window_height-self.player.step)):
            return True
        return False

    def on_loop(self):
        self.player.update()
        # print(list(zip(self.player.x, self.player.y)))

        # does snake eat apple?
        if self.is_collision(self.apple.x, self.apple.y,
                             self.player.x[0], self.player.y[0]):
            self.player.x = self.player.x + [self.player.x[-1]]
            self.player.y = self.player.y + [self.player.y[-1]]
            print(list(zip(self.player.x, self.player.y)))
            self.player.length = self.player.length + 1

            self.apple = Apple(randint(0, self.window_width / self.player.step - 1),
                               randint(0, self.window_height / self.player.step - 1))
            while (self.apple.x, self.apple.y) in list(zip(self.player.x, self.player.y)):
                self.apple = Apple(randint(0, self.window_width / self.player.step - 1),
                                   randint(0, self.window_height / self.player.step - 1))

        # does snake collide with itself?
        for i in range(2, self.player.length - 1):
            #             print(i,list(zip(self.player.x,self.player.y)))
            hit_self = self.is_collision(self.player.x[0],
                                         self.player.y[0],
                                         self.player.x[i],
                                         self.player.y[i])

            if hit_self:
                print(self.player.x[0], self.player.y[0])
                print(self.player.x[i], self.player.y[i])

                print("You lose! Hit yourself: ")
                print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
                print("x[" + str(i) + "] (" + str(self.player.x[i]) + "," + \
                      str(self.player.y[i]) + ")")
                #                 print('hit yourself')
                self.player = Snake()

                self.apple = Apple(randint(0, self.window_width / self.player.step - 1),
                                   randint(0, self.window_height / self.player.step - 1))

                break

        if self.out_of_bounds(self.player.x[0], self.player.y[0]):
            print("You lose! Hit wall: ")
            print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
            print("x[" + str(i) + "] (" + str(self.player.x[i]) + "," + \
                  str(self.player.y[i]) + ")")
            print('hit wall')
            self.player = Snake()

            self.apple = Apple(randint(0, self.window_width / self.player.step - 1),
                               randint(0, self.window_height / self.player.step - 1))
            while (self.apple.x, self.apple.y) in list(zip(self.player.x, self.player.y)):
                self.apple = Apple(randint(0, self.window_width / self.player.step - 1),
                                   randint(0, self.window_height / self.player.step - 1))

        pass


class App():

    def __init__(self, game=Game()):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.game = game

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.game.window_width,
                                                      self.game.window_height),
                                                     pygame.HWSURFACE)

        pygame.display.set_caption('Pygame pythonspot.com example')
        self._running = True
        self._image_surf = pygame.image.load("./images/snake1.png").convert()
        self._apple_surf = pygame.image.load("./images/apple_img.png").convert()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.game.player.draw(self._display_surf, self._image_surf)
        self.game.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            action = randint(0,3)

            if (keys[K_RIGHT]):
            # if action == 0:
                if self.game.player.direction == 1:
                    self.game.player.moveLeft()
                else:
                    self.game.player.moveRight()

            if (keys[K_LEFT]):
            # if action == 1:
                if self.game.player.direction == 0:
                    self.game.player.moveRight()
                else:
                    self.game.player.moveLeft()

            if (keys[K_UP]):
            # if action == 2:
                if self.game.player.direction == 3:
                    self.game.player.moveDown()
                else:
                    self.game.player.moveUp()

            if (keys[K_DOWN]):
            # if action == 3:
                if self.game.player.direction == 2:
                    self.game.player.moveUp()
                else:
                    self.game.player.moveDown()

            # if (keys[K_ESCAPE]):
            #     self._running = False

            self.game.on_loop()
            self.on_render()

            time.sleep(50.0 / 1000.0);
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
