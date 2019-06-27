# Reinforcement Learning for Snake

The code in this repository is used to train agents to play snake using techniques from reinforcement learning.
Tabular Q-learning and Value Function Approximation are implemented (using a simple Neural Network).

![](/images/demo.gif)

## Prerequisites

- [pygame](https://www.pygame.org/wiki/GettingStarted)
   - `python3 -m pip install -U pygame --user`
   
## Setup

- Clone the repository: `https://github.com/areevesman/reinforcement-learning-for-snake.git`
- `cd code`
- To play snake:
   - `python snake.py`
- To train a snake with Q-learning:
   - `python train_q_learning.py`
- To train a snake with VFA:
   - `python train_deep_q_learning.py`
- Training output will be saved to a `results` folder in the working directory

## Authors

- Adam Reevesman
- Evan Liu

## Acknowledgments

- Snake with pygame [tutorial](https://pythonspot.com/snake-with-pygame/)
- [LearnSnake: Teaching an AI to play Snake](https://italolelis.com/snake)
- [Designing AI: Solving Snake with Evolution](https://becominghuman.ai/designing-ai-solving-snake-with-evolution-f3dd6a9da867)
