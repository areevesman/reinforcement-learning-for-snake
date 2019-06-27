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
    def __init__(self, step=50, direction=0, length=5):
        self.step = step
        self.direction = direction
        self.length = length
        self.x = [(i + 1) * self.step for i in range(self.length - 1, -1, -1)]
        self.y = [1 * self.step for i in range(self.length - 1, -1, -1)]
        self.update_count_max = 2
        self.update_count = 0

    def update_snake(self):
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

    def update(self):
        self.update_count = self.update_count + 1
        if self.update_count > self.update_count_max:
            self.update_snake()
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
        self.window_width = window_width
        self.window_height = window_height
        self.snake = player
        self.apple = apple

    def is_collision(self, x1, y1, x2, y2):
        if ((x1 == x2) and (y1 == y2)):
            return True
        return False

    def is_out_of_bounds(self, x1, y1):
        if ((x1 < 0) or (y1 < 0) or
                (x1 > self.window_width-self.snake.step) or (y1 > self.window_height - self.snake.step)):
            return True
        return False

    def place_apple(self):
        # place apple
        self.apple = Apple(randint(0, self.window_width / self.snake.step - 1),
                           randint(0, self.window_height / self.snake.step - 1))
        # check for collision with snake
        while (self.apple.x, self.apple.y) in list(zip(self.snake.x, self.snake.y)):
            self.apple = Apple(randint(0, self.window_width / self.snake.step - 1),
                               randint(0, self.window_height / self.snake.step - 1))

    def try_eat_apple(self):
        # if snake finds apple
        if self.is_collision(self.apple.x,
                             self.apple.y,
                             self.snake.x[0],
                             self.snake.y[0]):
            # update snake
            self.snake.x = self.snake.x + [self.snake.x[-1]]
            self.snake.y = self.snake.y + [self.snake.y[-1]]
            self.snake.length = self.snake.length + 1
            print(f'Score: {self.snake.length}')

            self.place_apple()

    def try_hit_self(self):
        # does snake collide with itself?
        for i in range(2, self.snake.length - 1):
            if self.is_collision(self.snake.x[0],
                                 self.snake.y[0],
                                 self.snake.x[i],
                                 self.snake.y[i]):

                print(self.snake.x[0], self.snake.y[0])
                print(self.snake.x[i], self.snake.y[i])

                print("You lose! Hit yourself: ")
                print("x[0] (" + str(self.snake.x[0]) + "," + str(self.snake.y[0]) + ")")
                print("x[" + str(i) + "] (" + str(self.snake.x[i]) + "," + \
                      str(self.snake.y[i]) + ")")

                # should change this to a random snake position in the future
                self.snake = Snake()
                self.place_apple()
                break

    def try_out_of_bounds(self):
        if self.is_out_of_bounds(self.snake.x[0], self.snake.y[0]):
            print("You lose! Hit wall: ")
            print("x[0] (" + str(self.snake.x[0]) + "," + str(self.snake.y[0]) + ")")
            self.snake = Snake()
            self.place_apple()
        pass

    def make_move(self):
        self.snake.update()
        self.try_eat_apple()
        self.try_hit_self()
        self.try_out_of_bounds()


class App():
    def __init__(self, game=Game(), caption='Playing snake'):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.game = game
        self.caption = caption

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.game.window_width,
                                                    self.game.window_height),
                                                    pygame.HWSURFACE)

        pygame.display.set_caption('Time to learn!')
        self._running = True
        self._image_surf = pygame.image.load("../images/snake2.png").convert()
        self._apple_surf = pygame.image.load("../images/apple.png").convert()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.game.snake.draw(self._display_surf, self._image_surf)
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
                if self.game.snake.direction == 1:
                    self.game.snake.moveLeft()
                else:
                    self.game.snake.moveRight()

            if (keys[K_LEFT]):
                if self.game.snake.direction == 0:
                    self.game.snake.moveRight()
                else:
                    self.game.snake.moveLeft()

            if (keys[K_UP]):
                if self.game.snake.direction == 3:
                    self.game.snake.moveDown()
                else:
                    self.game.snake.moveUp()

            if (keys[K_DOWN]):
                if self.game.snake.direction == 2:
                    self.game.snake.moveUp()
                else:
                    self.game.snake.moveDown()

            if (keys[K_ESCAPE]):
                self._running = False

            self.game.make_move()
            self.on_render()

            time.sleep(20.0 / 1000.0)
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App(caption='Playing Snake')
    theApp.on_execute()
