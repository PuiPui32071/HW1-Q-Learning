import json
import random

import numpy as np
import pickle
from typing import Tuple, List


class QTable:
    def __init__(self,
                 state_bins: Tuple[int, int, int] = (10, 10, 10),
                 state_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
                 (0, 40), (0, 20), (0, 20)),
                 actions: Tuple[int] = (-40, -32, -24, -16, -8, 0, 8, 16, 24, 32, 40)):
        self.state_bins = state_bins
        self.state_ranges = state_ranges
        self.actions = actions
        self.num_actions = len(actions)

        # 初始化Q-Table
        self.q_table = np.random.rand(*state_bins, self.num_actions)

    def _discretize_state(self, state: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """將連續狀態轉為離散狀態 index"""
        discrete_state = []
        for i, (value, (low, high), bins) in enumerate(zip(state, self.state_ranges, self.state_bins)):
            value = min(max(value, low), high)  # clip範圍內
            bin_width = (high - low) / bins
            index = int((value - low) / bin_width)
            if index == bins:  # 邊界條件
                index -= 1
            discrete_state.append(index)
        return tuple(discrete_state)

    def _action_index(self, action: int) -> int:
        """取得 action 的 index"""
        return self.actions.index(action)

    def get_q_value(self, state: Tuple[float, float, float], action: int) -> float:
        discrete_state = self._discretize_state(state)
        action_idx = self._action_index(action)
        return self.q_table[(*discrete_state, action_idx)]

    def set_q_value(self, state: Tuple[float, float, float], action: int, value: float):
        discrete_state = self._discretize_state(state)
        action_idx = self._action_index(action)
        self.q_table[(*discrete_state, action_idx)] = value

    def get_best_action(self, state: Tuple[float, float, float]) -> int:
        discrete_state = self._discretize_state(state)
        best_action_index = np.argmax(self.q_table[discrete_state])
        return self.actions[best_action_index]

    def update_q_value(self, state: Tuple[float, float, float], action: int, reward: float,
                       next_state: Tuple[float, float, float], alpha: float, gamma: float):
        current_q = self.get_q_value(state, action)
        max_next_q = np.max(self.q_table[self._discretize_state(next_state)])
        new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_next_q)
        self.set_q_value(state, action, new_q)

    def save(self, filename: str):
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename: str):
        with open(f'{filename}.pkl', 'rb') as f:
            self.q_table = pickle.load(f)


class QLearner:
    def __init__(self, learning_rate=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = QTable()
        self.actions = self.q_table.actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get_q_value(state, action)

    def predict(self, state):
        # ε-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.q_table.get_best_action(state)

    def update(self, state, action, reward, next_state):
        self.q_table.update_q_value(state=state,
                                    action=action,
                                    reward=reward,
                                    next_state=next_state,
                                    alpha=self.lr,
                                    gamma=self.gamma)

    def reset(self):
        self.q_table = QTable(actions=self.actions)

    def load_qtable(self, filename: str):
        self.q_table.load(filename)
        with open(f'meta_{filename}.json', 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
            self.lr = data['lr']
            self.gamma = data['gamma']
            self.epsilon = data['epsilon']

    def save_qtable(self, filename: str):
        self.q_table.save(filename)
        with open(f'meta_{filename}.json', 'w', encoding='utf-8') as f:
            json.dump({
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f, ensure_ascii=False, indent=4)
