from random import randint
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Dense

from environment import *
from agent import *
from app_utils import *
from experience_replay import *


class TrainingApp(App):
    def train_model(self, model, n_episodes=100, epsilon=.1, alpha=.1, discount=.9, save_results=True, save_every=25):
        """train a neural network with experience replay
        examine and save results at every n_deaths_save"""
        self.on_init()
        # Training variables
        exp_replay = ExperienceReplay(max_memory=10_000, alpha=alpha, discount=discount)
        history = []
        moves_per_apple_per_episode = []
        loss = float('inf')
        apple_count = 0
        total_moves = 0
        apples_per_epsiode = []
        t0 = time.time()

        if save_results:
            if not os.path.exists('./results/deep_q_results'):
                os.makedirs('./results/deep_q_results')

        for e in range(n_episodes):
            apple_episode_count = 0
            move_count = 0
            # The next new episode.
            self.agent.reset_game()
            pygame.event.pump()
            self.on_render()
            over = self.agent.lost_game()
            score = len(self.agent.x)
            pygame.display.set_caption(f'{self.caption[:-2]} {score}')
            while not over:
                pygame.event.pump()
                self.on_render()
                # Get initial input
                current_state = self.agent.get_state()
                q = model.predict(current_state)
                if (np.random.rand() <= epsilon) and np.sum(current_state[:2] < 2):
                    new_direction = np.random.randint(0, 3, size=1)[0]
                    if (e % save_every) == 0:
                        print(self.agent.print_board())
                        print('moving randomly')
                else:
                    new_direction = np.argmax(q[0])
                    if (e % save_every) == 0:
                        print(self.agent.print_board())
                        print(q)

                # Apply action, get rewards and new state.
                future_state, reward = self.agent.move_snake(new_direction)
                pygame.event.pump()
                self.on_render()
                move_count = move_count + 1
                if reward == .5:
                    score = score + 1
                    pygame.display.set_caption(f'{self.caption[:-2]} {score}')
                    apple_episode_count = apple_episode_count + 1

                # Store experience.
                over = self.agent.lost_game()
                exp_replay.remember([current_state, new_direction, reward, future_state], over)

                # Get collected data to train model.
                inputs, targets = exp_replay.get_batch(model, batch_size=50)

                # Train model on experiences.
                loss = model.train_on_batch(inputs, targets)
                history.append(loss)
                total_moves = total_moves + 1

                if apple_episode_count == 0: 
                    moves_per_apple_per_episode.append(0)    
                else:
                    moves_per_apple_per_episode.append(move_count / apple_episode_count)

            if (e > 0) and (e % save_every == 0):
                t1 = time.time()
                time_delta = t1 - t0

                print(f"{time_delta/60:.2f} minutes")
                print(f"Epoch {e:,}/{n_episodes:,} | Loss {loss:.3f} | Moves per Game: {total_moves/e:.2f} | Apple count {apple_count}")
                apple_count = 0

                if save_results:
                    # save model and and plot various results to local directory
                    # serialize model to JSON
                    model_json = model.to_json()
                    with open(f"./results/deep_q_results/model-{e}.json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    model.save_weights(f"./results/deep_q_results/model-{e}.h5")
                    print(f"Saved model to disk")

                    # plot results
                    fig = plt.figure(figsize=(12, 7))
                    plt.plot(history)
                    plt.savefig(f'./results/deep_q_results/loss_per_move-{e}.png')
                    plt.close()
                    fig = plt.figure(figsize=(12, 7))
                    plt.plot(apples_per_epsiode, '.')
                    plt.savefig(f'./results/deep_q_results/score_per_episode-{e}.png')
                    plt.close()
            apples_per_epsiode.append(apple_episode_count)


def create_keras_model():
    model = Sequential()
    # add model layers
    model.add(Dense(5, input_shape=(6,)))
    model.add(Dense(3))
    # compile model
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == "__main__":
    theApp = TrainingApp(agent=Snake(), caption='Deep Q-Learning - Score: 0')
    model = create_keras_model()
    theApp.train_model(model, n_episodes=301, epsilon=.1, alpha=.1, discount=.9, save_results=True, save_every=25)
