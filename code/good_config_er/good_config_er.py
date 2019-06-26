from pygame.locals import *
from random import randint
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

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

    def place_apple(self):
        # place apple
        self.apple = Apple(randint(0, self.window_width / self.apple.step - 1),
                           randint(0, self.window_height / self.apple.step - 1))

    def apple_location(self):
        return [self.apple.x, self.apple.y]


class Snake():
    def __init__(self, game=Game(), step=50, initial_length=5):
        self.game = game
        self.step = step
        self.initial_length = initial_length
        self.x = None
        self.y = None
        self.direction = None
        self.snake_apple_xy_dir_to_apple_states = {
            # move up
            '0,1,1,1': [1, 1], '0,1,1,-1': [1, 0], '0,1,-1,-1': [0, 0], '0,1,-1,1': [0, 1],
            '0,1,1,0': [1, 0], '0,1,0,-1': [0, 0], '0,1,-1,0': [0, 0], '0,1,0,1': [0, 1],
            # move down
            '0,-1,1,1': [0, 0], '0,-1,1,-1': [0, 1], '0,-1,-1,-1': [1, 1], '0,-1,-1,1': [1, 0],
            '0,-1,1,0': [0, 0], '0,-1,0,-1': [0, 1], '0,-1,-1,0': [1, 0], '0,-1,0,1': [0, 0],
            # move right
            '1,0,1,1': [0, 1], '1,0,1,-1': [1, 1], '1,0,-1,-1': [1, 0], '1,0,-1,1': [0, 0],
            '1,0,1,0': [0, 1], '1,0,0,-1': [1, 0], '1,0,-1,0': [0, 0], '1,0,0,1': [0, 0],
            # move left
            '-1,0,1,1': [1, 0], '-1,0,1,-1': [0, 0], '-1,0,-1,-1': [0, 1], '-1,0,-1,1': [1,1],
            '-1,0,1,0': [0, 0], '-1,0,0,-1': [0, 0], '-1,0,-1,0': [0, 1], '-1,0,0,1': [1, 0]}
        self.snake_in_game_dir_xy_dir_to_xy_steps = {
            # move up
            '0,1,1': [-1, 0], '0,1,2': [1, 0], '0,1,0': [0, 1],
            # move down
            '0,-1,1': [1, 0], '0,-1,2': [-1, 0], '0,-1,0': [0, -1],
            # move right
            '1,0,1': [0, 1], '1,0,2': [0, -1], '1,0,0': [1, 0],
            # move left
            '-1,0,1': [0, -1], '-1,0,2': [0, 1], '-1,0,0': [-1,0]}

    def draw(self, surface, image,head):
        surface.blit(head, (self.x[0], self.y[0]))
        for i in range(1, len(self.x)):
            surface.blit(image, (self.x[i], self.y[i]))

    def lost_game(self):
        snake_postion = self.snake_location()
        if (self.tail_collision(snake_postion)) or (self.is_out_of_bounds(snake_postion)):
            return True
        elif self.apple_collision(snake_postion):
            return True
        else:
            return False

    def reset_game(self):
        p = randint(0, 3)
        constant = randint(0, int((self.game.window_width - self.step) / self.step))
        if p == 0:
            self.x = [(i + 1) * self.step for i in range(self.initial_length - 1, -1, -1)]
            self.y = [constant * self.step for i in range(self.initial_length - 1, -1, -1)]
        if p == 1:
            self.x = [constant * self.step for i in range(self.initial_length - 1, -1, -1)]
            self.y = [(i + 1) * self.step for i in range(self.initial_length - 1, -1, -1)]
        if p == 2:
            self.x = [(i + 4) * self.step for i in range(0, self.initial_length, 1)]
            self.y = [constant * self.step for i in range(self.initial_length - 1, -1, -1)]
        if p == 3:
            self.x = [constant * self.step for i in range(self.initial_length - 1, -1, -1)]
            self.y = [(i + 4) * self.step for i in range(0, self.initial_length, 1)]
        self.direction = randint(0, 2)
        self.request_apple()
        self.request_apple()

    def snake_location(self):
        return [[x,y] for x, y in zip(self.x, self.y)]

    def request_apple(self):
        self.game.place_apple()
        # check for collision with snake
        while self.game.apple_location() in self.snake_location():
            self.game.place_apple()

    def apple_collision(self, snake_position):
        if np.all(snake_position[0] == self.game.apple_location()):
            return True
        else:
            return False

    def tail_collision(self, snake_position):
        for i in range(2, len(snake_position)):
            if np.all(snake_position[0] == snake_position[i]):
                return True
        return False

    def is_out_of_bounds(self, snake_position):
        too_left = (snake_position[0][0] < 0)
        too_right = (snake_position[0][0] > self.game.window_width - self.step)
        too_low = (snake_position[0][1] > self.game.window_height - self.step)
        too_high = (snake_position[0][1] < 0)
        if too_left or too_right or too_low or too_high:
            return True
        return False

    def get_2d_snake_direction(self):
        snake_position = self.snake_location()
        direction = np.array(snake_position[0]) - np.array(snake_position[1])
        direction_vector = [int(direction[0]/self.step), int(direction[1]/self.step)]
        return direction_vector

    def food_direction(self):
        food_position = self.game.apple_location()
        snake_dir = self.get_2d_snake_direction()
        snake_head = self.snake_location()[0]
        food_dir = [int(np.sign(x / self.step)) for x in np.array(food_position) - np.array(snake_head)]
        snake_food_key =  str(snake_dir[0])+',' +str(snake_dir[1])+',' +  str(food_dir[0])+','+str(food_dir[1])
        food_right = self.snake_apple_xy_dir_to_apple_states[snake_food_key][0]
        food_straight = self.snake_apple_xy_dir_to_apple_states[snake_food_key][1]
        return food_right,food_straight

    def move_snake(self, new_direction):
        direction_vector = self.get_2d_snake_direction()
        update_snake_dir_key = str(direction_vector[0])+','+str(direction_vector[1])+','+ str(new_direction)
        tail = [self.x[-1],self.y[-1]]
        # update tail
        for i in range(len(self.x) - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]
        # update head
        self.x[0] = self.x[0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][0] * self.step
        self.y[0] = self.y[0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][1] * self.step
        if self.tail_collision(self.snake_location()):
            reward = -1
        elif self.is_out_of_bounds(self.snake_location()):
            reward = -1
        elif self.apple_collision(self.snake_location()):
            self.x.append(tail[0])
            self.y.append(tail[1])
            reward = 1
            self.request_apple()
        else:
            reward = -0.05
        new_state = self.get_state()
        return new_state, reward

    def is_safe(self, new_direction):
        snake_position = self.snake_location()
        direction_vector = self.get_2d_snake_direction()
        update_snake_dir_key = str(direction_vector[0]) + ',' + str(direction_vector[1]) + ',' + str(new_direction)

        x = [snake_position[i][0] for i in range(len(snake_position))]
        y = [snake_position[i][1] for i in range(len(snake_position))]
        # update tail
        for i in range(len(x) - 1, 0, -1):
            x[i] = x[i - 1]
            y[i] = y[i - 1]

        x[0] = x[0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][0] * self.step
        y[0] = y[0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][1] * self.step
        snake_position = [[x_, y_] for x_, y_ in zip(x, y)]

        # snake_position[1:] = [[x_,y_] for x_,y_ in zip(x,y)]
        # snake_position[0][0] = snake_position[0][0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][0] * self.step
        # snake_position[1][0] = snake_position[1][0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][1] * self.step
        if self.tail_collision(snake_position):
            return False
        elif self.is_out_of_bounds(snake_position):
            return False
        else:
            return True

    def get_state(self):
        is_safe_straight = self.is_safe(0)
        is_safe_left = self.is_safe(1)
        is_safe_right = self.is_safe(2)
        is_food_right = self.food_direction()[0]
        is_food_straight = self.food_direction()[1]
        state = [int(is_safe_straight),
                 int(is_safe_left),
                 int(is_safe_right),
                 int(is_food_right),
                 int(is_food_straight)]
        return np.array([state])

    def get_state2(self):
        snake_position = self.snake_location()
        apple_position = self.game.apple_location()

        # initial state is zeros,
        state = np.zeros((int(self.game.window_width / self.step),
                          int(self.game.window_height / self.step)),
                         dtype=int)

        # update with apple location
        x, y = 450-apple_position[1], apple_position[0]
        state[int(x / self.step), int(y / self.step)] = -1
        # update with snake head location
        x, y = 450-snake_position[0][1], snake_position[0][0]
        if (x < self.game.window_width) and (y < self.game.window_height):
            state[int(x / self.step), int(y / self.step)] = 2
        # update with snake body part locations
        for i in range(1, len(snake_position)):
            x, y = 450-snake_position[i][1], snake_position[i][0]
            state[int(x / self.step), int(y / self.step)] = 1
        return [[state]]


class ExperienceReplay():
    """Store the agent's experiences in order to collect enough example to get a reward signal"""
    def __init__(self, max_memory=10_000, alpha=.1, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.alpha = alpha
        self.good_memory = []

    def remember(self, states, game_over):
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.outputs[0].shape[1].value  # Read from neural network model
        env_dim = model.inputs[0].shape[1]  # Read from neural network model
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i:i + 1] = state_t[0]
            targets[i] = model.predict(state_t)[0]
            q_sa = model.predict(state_tp1).max()  # Find best model prediction for state_tp1
            if game_over:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = \
                    targets[i, action_t] + \
                    self.alpha * (reward_t + self.discount * q_sa - targets[i, action_t])  # Update with Q-learning
        return inputs, targets


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
        self._head_surf = pygame.image.load("../images/snake2.png").convert()
        self._image_surf = pygame.image.load("../images/snake2.png").convert()
        self._apple_surf = pygame.image.load("../images/apple_img.png").convert()

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.agent.draw(self._display_surf, self._image_surf,self._head_surf)
        self.agent.game.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def train_model(self, model, n_episodes=100, epsilon=.1):
        self.on_init()
        # Initialize experience replay object
        exp_replay = ExperienceReplay()
        # Training variables
        history = []
        loss = float('inf')
        apple_count = 0
        total_moves = 0
        apples_per_epsiode = []
        t0 = time.time()

        for e in range(n_episodes):
            apple_episode_count = 0
            # The next new episode.
            self.agent.reset_game()
            pygame.event.pump()
            self.on_render()
            over = self.agent.lost_game()
            while not over:
                pygame.event.pump()
                self.on_render()
                # Get initial input
                s_ = [[int(x/self.agent.step), int(y/self.agent.step)] for [x,y] in self.agent.snake_location()]
                current_state = self.agent.get_state()
                current_state_ = self.agent.get_state2()[0][0]
                q = model.predict(current_state)
                if np.random.rand() <= epsilon:
                    new_direction = np.random.randint(0, 3, size=1)[0]
                    if (e % 10) == 0:
                        print(current_state)
                        print('^random')
                else:
                    new_direction = np.argmax(q[0])
                    if (e % 10) == 0:
                        print(current_state)
                        print(q)

                # Apply action, get rewards and new state.
                future_state, reward = self.agent.move_snake(new_direction)
                pygame.event.pump()
                self.on_render()
                if reward == 1:
                    apple_count = apple_count + 1
                    apple_episode_count = apple_episode_count + 1

                # Store experience.
                over = self.agent.lost_game()
                # if over:
                #     print(self.agent.game.apple_location())
                #     print(s_)
                #     print(current_state_)
                #     print(current_state, new_direction, future_state)
                #     self.agent.move_snake(new_direction)
                #     print(self.agent.get_state2()[0][0])
                #     print(reward)

                # q_sa = model.predict(future_state).max()
                # td_error = np.abs(q_sa - q[0].argmax()) +.0001
                # print(td_error)
                # exp_replay.remember([current_state, new_direction, reward, future_state], over, q_sa, td_error)
                exp_replay.remember([current_state, new_direction, reward, future_state], over)

                # Get collected data to train model.
                inputs, targets = exp_replay.get_batch(model, batch_size=50)

                # Train model on experiences.
                loss = model.train_on_batch(inputs, targets)
                history.append(loss)
                total_moves = total_moves + 1

            if (e > 0) and (e % 100 == 0):
                t1 = time.time()
                time_delta = t1 - t0
                print(f"{time_delta/60:.2f} minutes")
                print(
                    f"Epoch {e:,}/{n_episodes:,} | Loss {loss:.3f} | Moves per Game: {total_moves/e:.2f} | Apple count {apple_count}")
                apple_count = 0

            # save model
            if (e > 0) and (e % 100 == 0):
                # serialize model to JSON
                model_json = model.to_json()
                with open(f"model-{e}.json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(f"model-{e}.h5")
                print(f"Saved model to disk")
                fig = plt.figure(figsize=(12, 7))
                plt.plot(history)
                plt.savefig(f'./loss_per_move-{e}.png')
                fig = plt.figure(figsize=(12, 7))
                plt.plot(apples_per_epsiode, '.')
                plt.savefig(f'./score_per_episode-{e}.png')
            apples_per_epsiode.append(apple_episode_count)


def create_keras_model():
    # create model
    model = Sequential()
    # add model layers
    model.add(Dense(5, input_shape=(5,)))
    # model.add(Dense(5, activation='relu'))
    # model.add(Dense(5, activation='relu'))
    # model.add(Dense(5, activation='relu'))
    # model.add(Dense(5, activation='relu'))
    model.add(Dense(3))
    # compile model
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == "__main__":
    # print(Snake().get_state()[0][0])
    theApp = App()
    model = create_keras_model()
    theApp.train_model(model, n_episodes=2_001)
