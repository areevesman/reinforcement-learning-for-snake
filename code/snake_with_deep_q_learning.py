from pygame.locals import *
from random import randint
import pygame
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout

import matplotlib.pyplot as plt

class Apple():
    def __init__(self, x=5, y=5, step=50):
        self.step = step
        self.x = x * self.step
        self.y = y * self.step

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))


class Game():
    def __init__(self, window_width=500, window_height=500, apple=Apple()):
        self.window_width = window_width
        self.window_height = window_height
        self.apple = apple
        self.apple_count = 0

    def place_apple(self):
        # place apple
        self.apple = Apple(randint(0, self.window_width / self.apple.step - 1),
                           randint(0, self.window_height / self.apple.step - 1))

    def apple_location(self):
        return [self.apple.x, self.apple.y]


class Snake():
    def __init__(self, game=Game(), step=50, direction=randint(1,2), length=5):
        self.game = game
        self.step = step
        self.direction = direction
        self.length = length
        p = randint(0, 3)
        if p == 0:
            self.x = [(i + 1) * self.step for i in range(self.length - 1, -1, -1)]
            self.y = [5 * self.step for i in range(self.length - 1, -1, -1)]
        if p == 1:
            self.x = [5 * self.step for i in range(self.length - 1, -1, -1)]
            self.y = [(i + 1) * self.step for i in range(self.length - 1, -1, -1)]
        if p == 2:
            self.x = [(i + 4) * self.step for i in range(0, self.length, 1)]
            self.y = [5 * self.step for i in range(self.length - 1, -1, -1)]
        if p == 3:
            self.x = [5 * self.step for i in range(self.length - 1, -1, -1)]
            self.y = [(i + 4) * self.step for i in range(0, self.length, 1)]

        while self.game.apple_location() in [[x,y] for x,y in zip(self.x, self.y)]:
            self.game.place_apple()

    def draw(self, surface, image):
        for i in range(0, self.length):
            surface.blit(image, (self.x[i], self.y[i]))

    def reset_game(self):
        # should change this to a random snake position in the future
        self.length = 5
        p = randint(0,3)
        if p == 0:
            self.x = [(i + 1) * self.step for i in range(self.length - 1, -1, -1)]
            self.y = [5 * self.step for i in range(self.length - 1, -1, -1)]
        if p == 1:
            self.x = [5 * self.step for i in range(self.length - 1, -1, -1)]
            self.y = [(i + 1) * self.step for i in range(self.length - 1, -1, -1)]
        if p == 2:
            self.x = [(i + 1) * self.step for i in range(0, self.length, 1)]
            self.y = [5 * self.step for i in range(self.length - 1, -1, -1)]
        if p == 3:
            self.x = [5 * self.step for i in range(self.length - 1, -1, -1)]
            self.y = [(i + 1) * self.step for i in range(0, self.length, 1)]
        self.direction = randint(1,2)
        self.request_apple()

    def snake_location(self):
        return [[x,y] for x,y in zip(self.x, self.y)]

    def request_apple(self):
        self.game.place_apple()
        # check for collision with snake
        while self.game.apple_location() in self.snake_location():
            self.game.place_apple()

    def update_position(self, new_direction):
        snake_position_old = np.array(self.snake_location())
        direction_vector = snake_position_old[0] - snake_position_old[1]
        up = np.array([0,self.step])
        down = np.array([0,-self.step])
        left = np.array([-self.step,0])
        right = np.array([self.step, 0])

        # update previous positions
        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        snake_position_incomplete = np.array(self.snake_location())

        if new_direction == 0: # go straight
            if np.all(direction_vector == right):
                snake_position_incomplete[0] = snake_position_incomplete[0] + right
            elif np.all(direction_vector == left):
                snake_position_incomplete[0] = snake_position_incomplete[0] + left
            elif np.all(direction_vector == up):
                snake_position_incomplete[0] = snake_position_incomplete[0] + up
            elif np.all(direction_vector == down):
                snake_position_incomplete[0] = snake_position_incomplete[0] + down
        elif new_direction == 1: # go right
            if np.all(direction_vector == right):
                snake_position_incomplete[0] = snake_position_incomplete[0] + down
            elif np.all(direction_vector == left):
                snake_position_incomplete[0] = snake_position_incomplete[0] + up
            elif np.all(direction_vector == up):
                snake_position_incomplete[0] = snake_position_incomplete[0] + right
            elif np.all(direction_vector == down):
                snake_position_incomplete[0] = snake_position_incomplete[0] + left
        elif new_direction == 2:  # go left
            if np.all(direction_vector == right):
                snake_position_incomplete[0] = snake_position_incomplete[0] + up
            elif np.all(direction_vector == left):
                snake_position_incomplete[0] = snake_position_incomplete[0] + down
            elif np.all(direction_vector == up):
                snake_position_incomplete[0] = snake_position_incomplete[0] + left
            elif np.all(direction_vector == down):
                snake_position_incomplete[0] = snake_position_incomplete[0] + right

        self.x = [snake_position_incomplete[0][0]] + [p[0] for p in snake_position_incomplete[1:]]
        self.y = [snake_position_incomplete[0][1]] + [p[1] for p in snake_position_incomplete[1:]]
        # print(self.snake_location())
        # print(self.is_out_of_bounds())

    def change_direction_and_move_snake(self, new_direction):
        self.update_position(new_direction)
        if self.tail_collision():
            reward = -1
            self.request_apple()
        elif self.is_out_of_bounds():
            reward = -1
            self.request_apple()
        elif self.apple_collision():
            # update snake
            self.x = self.x + [self.x[-1]]
            self.y = self.y + [self.y[-1]]
            self.length = self.length + 1
            # new apple
            self.request_apple()
            reward = 1
        else:
            reward = 0.05
        new_state = self.get_state()
        return new_state, reward


    def tail_collision(self):
        snake_position = self.snake_location()
        for i in range(2, len(snake_position)):
            if snake_position[0] == snake_position[i]:
                return True
        return False

    def apple_collision(self):
        snake_position = self.snake_location()
        if snake_position[0] == self.game.apple_location():
            return True
        else:
            return False

    def is_out_of_bounds(self):
        too_left = (self.x[0] < 0)
        too_right = (self.x[0] > self.game.window_width-self.step)
        too_low = (self.y[0] > self.game.window_height - self.step)
        too_high = (self.y[0] < 0)
        if too_left or too_right or too_low or too_high:
            return True
        return False

    def random_move(self):
        new_direction = randint(0,2)
        return new_direction

    def take_random_action(self):
        old_direction = self.direction
        new_direction = self.random_move()
        self.change_direction_and_move_snake(new_direction)
        if self.lost_game():
            self.reset_game()

    def get_state(self):
        snake_position = self.snake_location()
        apple_position = self.game.apple_location()

        state = np.zeros((int(self.game.window_width / self.step),
                          int(self.game.window_height / self.step)),
                         dtype=int)

        # print([apple_position] + snake_position)
        for y in range(0, self.game.window_height, self.step):
            for x in range(0, self.game.window_width, self.step):
                if [y, x] == [self.x[0], self.y[0]]:
                    state[int(x / self.step), int(y / self.step)] = 2
                elif [y,x] in snake_position:
                    state[int(x/self.step), int(y/self.step)] = 1
                elif [y, x] == apple_position:
                    state[int(x / self.step), int(y / self.step)] = -1
        # print(state.sum())
        return [[state]]

    def lost_game(self):
        if (self.tail_collision()) or (self.is_out_of_bounds()):
            return True
        elif self.apple_collision():
            return True
        else: return False


class ExperienceReplay():
    """Store the agent's experiences in order to collect enough example to get a reward signal"""
    def __init__(self, max_memory=100, alpha=.1, discount=1):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.alpha = alpha

    def remember(self, states, game_over):
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            # if states[2] != 1:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)  # Given to you
        num_actions = model.outputs[0].shape[1].value  # Read from neural network model
        env_dim = model.inputs[0].shape[1]  # Read from neural network model
        inputs = np.zeros((min(len_memory, batch_size), env_dim, env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]
            # print(state_t[0][0])
            # print(inputs)
            # print(model.inputs[0])
            # print(env_dim, inputs)
            inputs[i:i + 1] = state_t[0]
            targets[i] = model.predict(state_t)[0]
            q_sa = model.predict(state_tp1).max()  # Find best model prediction for state_tp1
            # print(q_sa)
            if game_over:  # Given to you
                targets[i, action_t] = reward_t
            else:  # Given to you
                targets[i, action_t] = \
                    targets[i, action_t] + \
                    (reward_t + self.alpha*(self.discount * q_sa - targets[i, action_t])) # Update with Q-learning

        return inputs, targets


def create_keras_model():
    # create model
    model = Sequential()
    # add model layers
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape = (10, 10)))
    model.add(Dropout(rate=.1))
    model.add(Conv1D(16, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def train_model(model, agent=Snake(), epsilon=.1):
    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=100)
    # Training variables
    n_episodes = 3000
    history = []
    loss = float('inf')
    apple_count = 0
    apples_per_epsiode = []

    for e in range(n_episodes):
        apple_episode_count = 0
        # The next new episode.
        agent.reset_game()
        over = agent.lost_game()
        while not over:

            # Get initial input
            current_state = agent.get_state()

            # Get next action with eplison-greedy.
            if np.random.rand() <= epsilon:
                new_direction = np.random.randint(0, 3, size=1)[0]
            else:
                q = model.predict(current_state)
                new_direction = np.argmax(q[0])

            # Apply action, get rewards and new state.
            future_state, reward = agent.change_direction_and_move_snake(new_direction)
            if reward == 1:
                apple_count += 1
                apple_episode_count = apple_episode_count + 1

            # Store experience.
            over = agent.lost_game()
            exp_replay.remember([current_state, new_direction, reward, future_state], over)

            # Get collected data to train model.
            inputs, targets = exp_replay.get_batch(model, batch_size=50)

            # Train model on experiences.
            loss = model.train_on_batch(inputs, targets)
            history.append(loss)

        if (e == 0) or (e % 10 == 0):
            print(f"Epoch {e:03d}/{n_episodes:,} | Loss {loss:.3f} | Win count {apple_count}")
            apple_count = 0
        apples_per_epsiode.append(apple_episode_count)

    fig = plt.figure(figsize=(12,7))
    plt.plot(history)
    plt.savefig('./loss_per_move.png')
    fig = plt.figure(figsize=(12, 7))
    plt.plot(apples_per_epsiode)
    plt.savefig('./score_per_episode.png')


class App():

    def __init__(self, agent=Snake()):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.agent = agent

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.agent.game.window_width,
                                                      self.agent.game.window_height),
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
        self.agent.draw(self._display_surf, self._image_surf)
        self.agent.game.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            pygame.event.pump()

            self.agent.take_random_action()
            self.on_render()

            time.sleep(100.0 / 1000.0)
        self.on_cleanup()

if __name__ == "__main__":
    # print(Snake().get_state()[0][0])
    # theApp = App()
    # theApp.on_execute()
    model = create_keras_model()
    train_model(model)
