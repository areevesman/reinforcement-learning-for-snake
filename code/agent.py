import numpy as np
from environment import *
from pygame.transform import rotate

class Snake():
    def __init__(self, game=Game(), step=50, initial_length=5):
        self.game = game
        self.step = step # number of pixels for consistency with pygame app
        self.initial_length = initial_length
        # call self.update_snake to update snake location and direction
        self.x = None
        self.y = None
        self.direction = None
        # key[:2] is snake direction and key[2:] is relative apple location,
        # value is whether or not apple is straight, left, right
        self.snake_apple_xy_dir_to_apple_states = {
            # move up
            '0,1,1,1': [1, 1, 0], '0,1,1,-1': [1, 0, 0], '0,1,-1,-1': [0, 0, 1], '0,1,-1,1': [0, 1, 1],
            '0,1,1,0': [1, 0,0], '0,1,0,-1': [0, 0, 0], '0,1,-1,0': [0, 0, 0], '0,1,0,1': [0, 1, 0],
            # move down
            '0,-1,1,1': [0, 0, 1], '0,-1,1,-1': [0, 1, 0], '0,-1,-1,-1': [1, 1, 0], '0,-1,-1,1': [1, 0, 0],
            '0,-1,1,0': [0, 0, 0], '0,-1,0,-1': [0, 1, 0], '0,-1,-1,0': [1, 0, 0], '0,-1,0,1': [0, 0, 0],
            # move right
            '1,0,1,1': [0, 1, 1], '1,0,1,-1': [1, 1, 0], '1,0,-1,-1': [1, 0, 1], '1,0,-1,1': [0, 0, 1],
            '1,0,1,0': [0, 1, 0], '1,0,0,-1': [1, 0, 0], '1,0,-1,0': [0, 0, 0], '1,0,0,1': [0, 0, 0],
            # move left
            '-1,0,1,1': [1, 0, 0], '-1,0,1,-1': [0, 0, 1], '-1,0,-1,-1': [0, 1, 1], '-1,0,-1,1': [1, 1, 0],
            '-1,0,1,0': [0, 0, 0], '-1,0,0,-1': [0, 0, 0], '-1,0,-1,0': [0, 1, 0], '-1,0,0,1': [1, 0, 0]}
        # key[:2] is snake direction and key[2] is new direction
        # value is direction vector to move snake
        self.snake_in_game_dir_xy_dir_to_xy_steps = {
            # move up
            '0,1,1': [-1, 0], '0,1,2': [1, 0], '0,1,0': [0, 1],
            # move down
            '0,-1,1': [1, 0], '0,-1,2': [-1, 0], '0,-1,0': [0, -1],
            # move right
            '1,0,1': [0, 1], '1,0,2': [0, -1], '1,0,0': [1, 0],
            # move left
            '-1,0,1': [0, -1], '-1,0,2': [0, 1], '-1,0,0': [-1,0]}

    def draw(self, surface, image, head):
        """draw in pygame app"""
        direction_vector = self.get_2d_snake_direction()
        if np.all(direction_vector == [1,0]):
            head = rotate(head, -90)
        elif np.all(direction_vector == [-1,0]):
            head = rotate(head, 90)
        elif np.all(direction_vector == [0,1]):
            head = rotate(head, 180)
        else:
            head = rotate(head, 0)
        surface.blit(head, (self.x[0], self.y[0]))
        for i in range(1, len(self.x)):
            surface.blit(image, (self.x[i], self.y[i]))

    def lost_game(self):
        """is the game over"""
        snake_postion = self.snake_location()
        if (self.tail_collision(snake_postion)) or (self.is_out_of_bounds(snake_postion)):
            return True
        elif self.apple_collision(snake_postion):
            return True
        else:
            return False

    def reset_game(self):
        """random snake location, direction, and apple"""
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
        """new apple, non-overlapping with snake"""
        self.game.place_apple()
        # check for collision with snake
        while self.game.apple_location() in self.snake_location():
            self.game.place_apple()

    def apple_collision(self, snake_position):
        """does snake eat apple"""
        if np.all(snake_position[0] == self.game.apple_location()):
            return True
        else:
            return False

    def tail_collision(self, snake_position):
        """does snake hit itself"""
        for i in range(2, len(snake_position)):
            if np.all(snake_position[0] == snake_position[i]):
                return True
        return False

    def is_out_of_bounds(self, snake_position):
        """does snake hit wall"""
        too_left = (snake_position[0][0] < 0)
        too_right = (snake_position[0][0] > self.game.window_width - self.step)
        too_low = (snake_position[0][1] > self.game.window_height - self.step)
        too_high = (snake_position[0][1] < 0)
        if too_left or too_right or too_low or too_high:
            return True
        return False

    def get_2d_snake_direction(self):
        """get direction of snake in terms of x,y: ex: [1,0] is right"""
        snake_position = self.snake_location()
        direction = np.array(snake_position[0]) - np.array(snake_position[1])
        direction_vector = [int(direction[0]/self.step), int(direction[1]/self.step)]
        return direction_vector

    def food_direction(self):
        """booleans for relative location of food"""
        food_position = self.game.apple_location()
        snake_dir = self.get_2d_snake_direction()
        snake_head = self.snake_location()[0]
        food_dir = [int(np.sign(x / self.step)) for x in np.array(food_position) - np.array(snake_head)]
        snake_food_key =  str(snake_dir[0])+',' +str(snake_dir[1])+',' +  str(food_dir[0])+','+str(food_dir[1])
        food_right = self.snake_apple_xy_dir_to_apple_states[snake_food_key][0]
        food_straight = self.snake_apple_xy_dir_to_apple_states[snake_food_key][1]
        food_left = self.snake_apple_xy_dir_to_apple_states[snake_food_key][2]
        return food_right, food_straight, food_left

    def move_snake(self, new_direction):
        """move snake based on a new direction, return new state and reward"""
        direction_vector = self.get_2d_snake_direction()
        update_snake_dir_key = str(direction_vector[0])+','+str(direction_vector[1])+','+ str(new_direction)
        tail = [self.x[-1], self.y[-1]]
        # update tail
        for i in range(len(self.x) - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]
        # update head
        self.x[0] = self.x[0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][0] * self.step
        self.y[0] = self.y[0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][1] * self.step
        # collect rewards
        if self.tail_collision(self.snake_location()):
            reward = -1
        elif self.is_out_of_bounds(self.snake_location()):
            reward = -1
        elif self.apple_collision(self.snake_location()):
            self.x.append(tail[0])
            self.y.append(tail[1])
            reward = .5
            self.request_apple()
        else:
            reward = -0.05
        new_state = self.get_state()
        return new_state, reward

    def is_safe(self, new_direction):
        """is it safe to move one step in the new direction"""
        snake_position = self.snake_location()
        direction_vector = self.get_2d_snake_direction()
        update_snake_dir_key = str(direction_vector[0]) + ',' + str(direction_vector[1]) + ',' + str(new_direction)
        # update tail
        x = [snake_position[i][0] for i in range(len(snake_position))]
        y = [snake_position[i][1] for i in range(len(snake_position))]
        for i in range(len(x) - 1, 0, -1):
            x[i] = x[i - 1]
            y[i] = y[i - 1]
        # update head
        x[0] = x[0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][0] * self.step
        y[0] = y[0] + self.snake_in_game_dir_xy_dir_to_xy_steps[update_snake_dir_key][1] * self.step
        snake_position = [[x_, y_] for x_, y_ in zip(x, y)]
        # check safety
        if self.tail_collision(snake_position):
            return False
        elif self.is_out_of_bounds(snake_position):
            return False
        else:
            return True

    def get_state(self):
        """get the state for the model"""
        is_safe_straight = self.is_safe(0)
        is_safe_left = self.is_safe(1)
        is_safe_right = self.is_safe(2)
        is_food_right = self.food_direction()[0]
        is_food_straight = self.food_direction()[1]
        is_food_left = self.food_direction()[2]

        state = (int(is_safe_straight),
                 int(is_safe_left),
                 int(is_safe_right),
                 int(is_food_right),
                 int(is_food_straight),
                 int(is_food_left))
        return np.array([state])

    def print_board(self):
        """show the board as x,y coordinates for testing and debugging"""
        snake_position = self.snake_location()
        apple_position = self.game.apple_location()

        # initial state is zeros,
        state = np.zeros((int(self.game.window_width / self.step),
                          int(self.game.window_height / self.step)),
                         dtype=int)

        # update with apple location
        i, j = self.game.window_height - self.step - apple_position[1], apple_position[0]
        state[int(i / self.step), int(j / self.step)] = -1
        # update with snake head location
        i, j = self.game.window_height - self.step - snake_position[0][1], snake_position[0][0]
        if (i < self.game.window_width) and (j < self.game.window_height):
            state[int(i / self.step), int(j / self.step)] = 2
        # update with snake body part locations
        for k in range(1, len(snake_position)):
            i, j = self.game.window_height - self.step - snake_position[k][1], snake_position[k][0]
            state[int(i / self.step), int(j / self.step)] = 1
        return state