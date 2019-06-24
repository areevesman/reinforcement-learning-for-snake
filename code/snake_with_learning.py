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
        self.y = [5 * self.step for i in range(self.length - 1, -1, -1)]

    def update(self):

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

    def __init__(self, window_width=500, window_height=500, player=Snake(), apple=Apple(), epsilon=.1, alpha=.1):
        self.window_width = window_width
        self.window_height = window_height
        self.snake = player
        self.apple = apple
        self.reward = 0
        self.epsilon = epsilon
        self.Q = {}
        self.alpha = alpha
        self.num_moves = 0
        self.apple_count = 0
        self.death_count = 0

        grid_locations = [(x, y) for x in range(self.window_width * -1, self.window_width + 50, 50)
                          for y in range(self.window_height * -1, self.window_height + 50, 50)]
        states = [(tail, fruit) for tail in grid_locations for fruit in grid_locations]
        actions = [i for i in range(0, 4)]
        state_actions = [(s, a) for s in states for a in actions]

        for sa in state_actions:
            self.Q[str(sa)] = 0

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
            self.apple_count = self.apple_count + 1
            self.reward = 1
            # print(f'Score: {self.snake.length}')

            self.place_apple()


    def try_hit_self(self):
        # does snake collide with itself?
        for i in range(2, self.snake.length):
            if self.is_collision(self.snake.x[0],
                                 self.snake.y[0],
                                 self.snake.x[i],
                                 self.snake.y[i]):

                self.reward = -1
                self.death_count = self.death_count + 1
                # should change this to a random snake position in the future
                self.snake = Snake()
                self.place_apple()
                break

    def try_out_of_bounds(self):
        if self.is_out_of_bounds(self.snake.x[0], self.snake.y[0]):
            # print("You lose! Hit wall: ")
            # print("x[0] (" + str(self.snake.x[0]) + "," + str(self.snake.y[0]) + ")")
            self.reward = -1
            self.death_count = self.death_count + 1
            self.snake = Snake()
            self.place_apple()

    def make_move(self):
        self.snake.update()
        self.try_eat_apple()
        self.try_hit_self()
        self.try_out_of_bounds()
        if self.reward == 0:
            self.reward = -0.05
        # print(self.reward)
        self.act_and_learn()
        self.reward = 0

    def get_state(self):
        state = ((self.snake.x[-1] - self.snake.x[0],
                   self.snake.y[-1] - self.snake.y[0]),
                  (self.apple.x - self.snake.x[0],
                   self.apple.y - self.snake.y[0]))
        return state

    def select_best_action(self):
        illegal_moves = {0: 1, 1: 0, 2: 3, 3: 2}
        current_state = self.get_state()
        possible_q_vals = []
        for i in range(4):
            possible_q_vals.append(((current_state,i),
                                     self.Q['('+str(current_state)+', '+str(i)+')']))

        for tup in possible_q_vals:
            if tup[0] == (current_state, illegal_moves[self.snake.direction]):
                possible_q_vals.remove(tup)
                break


        possible_q_vals = sorted(possible_q_vals, reverse=True, key=lambda x: x[1])
        max_tuple = possible_q_vals[0]
        max_action = max_tuple[0][1]
        max_value = max_tuple[1]

        return current_state, max_action, max_value

    def act_and_learn(self):
        current_state, best_next_action, best_next_state_value = self.select_best_action()

        p = randint(0, 100) / 100.0
        if p < self.epsilon:
            actual_next_action = randint(0, 3)
        else:
            actual_next_action = best_next_action

        old_value = self.Q['(' + str(current_state) + ', ' + str(best_next_action) + ')']
        td_target = self.reward + best_next_state_value
        self.Q['(' + str(current_state) + ', ' + str(best_next_action) + ')'] = old_value + self.alpha * (td_target - old_value)

        moves = {0: self.snake.moveRight,
                 1: self.snake.moveLeft,
                 2: self.snake.moveUp,
                 3: self.snake.moveDown}

        move = moves[actual_next_action]
        move()


class App():

    def __init__(self, game=Game()):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.game = game
        self.high_score = self.game.snake.length

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.game.window_width,
                                                      self.game.window_height),
                                                     pygame.HWSURFACE)

        pygame.display.set_caption('Time to learn!')
        self._running = True
        self._image_surf = pygame.image.load("../images/snake2.png").convert()
        self._apple_surf = pygame.image.load("../images/apple_img.png").convert()

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
        i = 0
        while (self._running):
            pygame.event.pump()

            self.game.make_move()
            # self.on_render()

            # time.sleep(1.0 / 1000.0)
            i = i + 1
            if self.game.snake.length > self.high_score:
                self.high_score = self.game.snake.length
            if i % 100 == 0:
                print(f'Total moves:  {i}')
                print(f'High score:   {self.high_score}')
                print(f'Total apples: {self.game.apple_count}')
                print(f'Apple pct:    {(self.game.apple_count / i*100):.3f}%')
                print(f'Death pct:    {(self.game.death_count / i*100):.3f}%')
                print(self.game.select_best_action())
                print()
            if i == 5000:
                break
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()