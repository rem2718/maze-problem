from collections import defaultdict

import gymnasium as gym
import numpy as np

class OnAgent:
    def __init__(self, env: gym.Env, epsilon=0.1, theta=15, first_visit=False, ars=False):
        self.env = env
        self.gamma = env.gamma
        self.epsilon = epsilon
        self.theta = theta
        self.first_visit = first_visit
        self.ars = ars
        self.episode = []
        self.stable_count = 0
        self.actions = list(range(env.action_space.n))
        self.policy = {}
        self.q_values = {}
        self.d = {}
        self.returns = defaultdict(list)
        self.values = np.full((self.env.size, self.env.size), np.nan)
        self.arrows = np.full((self.env.size, self.env.size), '-', dtype=str)
        
    def _q_to_v(self):
        value_dict = defaultdict(list)
        for (s, _), q in self.q_values.items():
            value_dict[s].append(q)
        for (x, y), q_list in value_dict.items():
            self.values[x, y] = np.mean(q_list) if len(q_list) != 0 else np.nan
            
    def _p_to_a(self):
        dirs = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        for (x, y), probs in self.policy.items():
            self.arrows[x, y] = dirs[np.nanargmax(probs)]
    
    def _V(self, s):
        v = self.q_values[s[0], s[1]]
        return 0.0 if np.isnan(v) else v
    
    def _policy(self, s):
        if s not in self.policy:
            self.policy[s] = [1.0 / len(self.actions) for _ in range(len(self.actions))]
        return self.policy[s]
    
    def _OnPolicyMCFV(self):
        visited = set()
        for t, (s, a, r) in enumerate(self.episode):
            pair = (s, a)
            if np.isnan(r) or pair in visited:
                continue
            
            visited.add(pair)
            R = 0
            for k in range(t, len(self.episode)):
                r = self.episode[k][2]
                if not np.isnan(r):
                    R += r * (self.gamma ** (k - t))           
            
            self.returns[pair].append(R)
            self.q_values[pair] = np.mean(self.returns[pair])
        self.episode = []
        return self._policyImprovement()
    
    def _OnPolicyMCEV(self):
        R = 0
        for s, a, r in reversed(self.episode):
            if np.isnan(r):
                continue
            pair = (s, a)
            R = r + self.gamma * R        
            self.returns[pair].append(R)
            self.q_values[pair] = np.mean(self.returns[pair])
        self.episode = []
        return self._policyImprovement()
        
    def _policyImprovement(self):
        policy_stable = True
        for s, old_action_probs in self.policy.items():
            new_action_probs = [self.epsilon/len(self.actions) for _ in range(len(self.actions))]
            if s == (0, 0): 
                continue 
            
            q_list = [self.q_values.get((s, a), np.nan) for a in self.actions]
            best_action = np.random.choice(self.actions) if np.all(np.isnan(q_list)) else np.nanargmax(q_list)
            new_action_probs[best_action] += (1.0 - self.epsilon)
            self.policy[s] = new_action_probs   
            if not np.allclose(old_action_probs, new_action_probs):
                policy_stable = False
        
        if policy_stable:
            self.stable_count += 1
            
        self._p_to_a() if self.ars else self._q_to_v()
        return self.stable_count >= self.theta
                
    def get_action(self, s):
        action_probs = self._policy(tuple(s))
        return np.random.choice(len(action_probs), p=action_probs)

    def update(self, s, a, r, nxt_s, over):
        self.episode.append((tuple(s), a, r))
        if over:
            return self._OnPolicyMCFV() if self.first_visit else self._OnPolicyMCEV()
        return False
    
    def get_values(self):
        return self.arrows if self.ars else self.values



class OffAgent:
    def __init__(self, env: gym.Env, theta=15, ars=False, train=True):
        self.env = env
        self.gamma = env.gamma
        self.theta = theta
        self.ars = ars
        self.train = train
        self.is_reached = False
        self.episode = []
        self.stable_count = 0
        self.actions = list(range(env.action_space.n))
        self.target_policy = {}
        self.behavior_policy = {}
        self.q_values = {}
        self.d = {}
        self.values = np.full((self.env.size, self.env.size), np.nan)
        self.arrows = np.full((self.env.size, self.env.size), '-', dtype=str)
        
    def _q_to_v(self):
        value_dict = defaultdict(list)
        for (s, _), q in self.q_values.items():
            value_dict[s].append(q)
        for (x, y), q_list in value_dict.items():
            self.values[x, y] = np.mean(q_list) if len(q_list) != 0 else np.nan
            
    def _p_to_a(self):
        dirs = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        for (x, y), probs in self.target_policy.items():
            self.arrows[x, y] = dirs[np.nanargmax(probs)]
    
    def _V(self, s):
        v = self.q_values[s[0], s[1]]
        return 0.0 if np.isnan(v) else v
    
    def _target_policy(self, s):
        if s not in self.target_policy:
            self.target_policy[s] = np.zeros(len(self.actions))
            best_action = np.random.choice(self.actions)
            self.target_policy[s][best_action] = 1.0
        return self.target_policy[s]
    
    def _behavior_policy(self, s):
        if s not in self.behavior_policy:
            self.behavior_policy[s] = np.full(len(self.actions), 1.0 / len(self.actions)) 
        return self.behavior_policy[s]    
    
    def _OffPolicyMC(self):
        R = 0
        W = 1
        for s, a, r in reversed(self.episode):
            if np.isnan(r):
                continue
            pair = (s, a)
            R = r + self.gamma * R
            self.d[pair] = self.d.get(pair, 0) + W
            self.q_values[pair] = self.q_values.get(pair, 0) + (W / self.d[pair]) * (R - self.q_values.get(pair, 0))
            if s == (self.env.size - 1, self.env.size - 1):
                self.is_reached = True
            if self._target_policy(s)[a] == 0:
                break
            W = W * 1 / self._behavior_policy(s)[a]
        self.episode = []
        return self._policyImprovement()
        
    def _policyImprovement(self):
        policy_stable = True
        for s, old_action_probs in self.target_policy.items():
            new_action_probs = np.zeros(len(self.actions))
            if s == (0, 0): 
                continue 
            
            q_list = [self.q_values.get((s, a), np.nan) for a in self.actions]
            best_action = np.random.choice(self.actions) if np.all(np.isnan(q_list)) else np.nanargmax(q_list)
            new_action_probs[best_action] = 1
            self.target_policy[s] = new_action_probs   
            if not np.allclose(old_action_probs, new_action_probs):
                policy_stable = False
        
        if self.is_reached and policy_stable:
            self.stable_count += 1
            
        self._p_to_a() if self.ars else self._q_to_v()
        return self.stable_count >= self.theta
    
    def get_action(self, s):
        action_probs = self._behavior_policy(tuple(s)) if self.train else self._target_policy(tuple(s))
        return np.random.choice(len(action_probs), p=action_probs)

    def update(self, s, a, r, nxt_s, over):
        self.episode.append((tuple(s), a, r))
        if over:
            return self._OffPolicyMC()
        return False
    
    def get_values(self):
        return self.arrows if self.ars else self.values

    
    
