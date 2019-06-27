import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from environment import *
from agent import *
from app_utils import *


class TrainingApp(App):
    def update_q_table(self, n_episodes=1000, alpha=.1, discount=.9, epsilon=.1, save_results=True, save_every=100):
        if self.on_init() == False:
            self._running = False

        if save_results:
            if not os.path.exists('./results/q_results'):
                os.makedirs('./results/q_results')

        total_moves = 0
        t0 = time.time()
        apples_per_episode = []
        apple_episode_count = 0
        for e in range(n_episodes):
            # The next new episode
            self.agent.reset_game()
            pygame.event.pump()
            self.on_render()
            over = self.agent.lost_game()
            while not over:
                pygame.event.pump()
                self.on_render()
                # Get state and q values for state action pairs
                current_state = tuple(self.agent.get_state()[0])
                q = [self.Q['(' + str(current_state) + ', ' + str(a) +')'] for a in range(3)]

                if (np.random.rand() <= epsilon):
                    new_direction = np.random.randint(0, 2, size=1)[0]
                    if (e % save_every) == 0:
                        print(self.agent.print_board())
                        print('moving randomly')
                else:
                    new_direction = np.argmax(q)
                    if (e % save_every) == 0:
                        print(self.agent.print_board())
                        print(q)

                # Apply action, get rewards and new state.
                future_state, reward = self.agent.move_snake(new_direction)
                pygame.event.pump()
                self.on_render()

                # Update with q learning
                old_value = self.Q['(' + str(current_state) + ', ' + str(new_direction) + ')']
                td_target = reward + discount*np.max(q)
                self.Q['(' + str(current_state) + ', ' + str(new_direction) + ')'] = old_value + alpha * (td_target - old_value)

                # Check for terminal state
                over = self.agent.lost_game()

                # if snake found apple, update apple_episode_count
                if reward == .5:
                    apple_episode_count = apple_episode_count + 1
                total_moves = total_moves + 1

            if (e > 0) and (e % save_every == 0):
                t1 = time.time()
                time_delta = t1 - t0

                print(f"{time_delta/60:.2f} minutes")
                print(f"Epoch {e:,}/{n_episodes:,} Moves per Game: {total_moves/e:.2f} | Apple count {apple_episode_count}")
                apple_episode_count = 0

                if save_results:
                    with open(f'./results/q_results/Q_vals-{e}.json', 'w') as file:
                        file.write(json.dumps(self.Q))
                    fig = plt.figure(figsize=(12, 7))
                    plt.plot(apples_per_episode, '.')
                    plt.savefig(f'./results/q_results/score_per_episode-{e}.png')
                    plt.close()
            apples_per_episode.append(apple_episode_count)
        self.on_cleanup()


if __name__ == "__main__":
    theApp = TrainingApp(agent=Snake(), caption='Q-Learning')
    theApp.update_q_table(n_episodes=301, alpha=.1, discount=.9, epsilon=.1, save_results=True, save_every=25)
