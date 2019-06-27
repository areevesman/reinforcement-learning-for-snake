import numpy as np

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
