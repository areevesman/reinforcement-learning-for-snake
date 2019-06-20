# Snake meets Reinforcement Learning

## Contributors

- Adam Reevesman
- Evan Liu

## Research question

Can we use Q-learning to train an agent to play Snake? Can we improve upon the results (lower training time and higher scores) with Value Function Approximation (VFA) and deep learning?

## Progress

The python script [code/snake.py](code/snake.py) contains the implementation.
- environment (see the `Game` class)
- agent (see the `Snake` class)
  - can take random actions or be controlled by human (see `App` class)

## Todo in order to finish
- add rewards to environment
  - large positive reward like +1 at apple location
  - large negative reward like -1 at walls and snake body
  - small negative rewards like -0.05 elsewhere to encourage snake to seek out apple
- add code for reinforcement learning (Q-learning/VFA)
