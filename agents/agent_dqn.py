import os
import sys
import torch
import random
import torch.nn as nn
import numpy as np

from agents.utils import Brain, Memory

np.random.seed(47)
random.seed(47)

class Agent_DQN(nn.Module):
    def __init__(self, player="Red", device=torch.device("cuda")):
        super(Agent_DQN, self).__init__()
        assert player in ("Red", "Blue"), "Wrong player color"
        self.id = player
        self.prioritized = True  # True = prioritized mempory replay, False = DQN_nature, https://blog.csdn.net/gsww404/article/details/103673852

        self.a_dim = 64
        self.s_dim = 64
        self.device = device
        self.gamma = 0.95  # reward decay rate
        self.alpha1 = 0.1  # soft copy weights from blue to red, alpha1 updates while (1-alpha1) remains
        self.alpha2 = 0.1  # soft copy weights from eval net to target net, alpha2 updates while (1-alpha2) remains
        self.epsilon_max = 1.0
        self.batch_size = 64
        self.epsilon_increment = None  # 0.0005
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
        # total learning step
        self.learn_step_counter = 0  # count how many times the eval net has been updated, used to set a basis for updating the target net
        self.replace_target_iter = 200

        self.brain_evl = Brain(a_dim=self.a_dim, s_dim=self.s_dim).to(device)
        if self.id == "Blue":  # only while player learns
            self.brain_tgt = Brain(a_dim=self.a_dim, s_dim=self.s_dim).to(device)
            self.memory_size = 10000
            self.memory_counter = 0
            # initialize zero memory [s, a, r, done, s_, a_cand] = 64+1+1+64+64
            if self.prioritized:  # prioritized experience replay
                self.memory = Memory(self.memory_size)
            else:
                self.memory = np.zeros((self.memory_size, self.s_dim + 1 + 1 + 1 + self.s_dim + self.a_dim), dtype=np.float)
            self.opt = torch.optim.Adam(self.brain_evl.parameters(), lr=1e-3, weight_decay=0.1)
            self.critic = nn.MSELoss()

    def choose_action(self, obs, a_possible):
        """
        :param obs: list[list], shape=[8, 8]
        :param a_possible: a set of tuples (row, col)
        :return: a tuple of (row, col)
        """
        self.brain_evl.eval()
        with torch.no_grad():
            obs = torch.tensor(np.array(obs).ravel(), dtype=torch.float32).unsqueeze(0).to(self.device)  # shape = (1, 64)
            mask = torch.tensor([[True] * 64], dtype=torch.bool).to(self.device)  # shape = (1, 64)
            for r, c in a_possible:
                mask[0][r * 8 + c] = False  # do not mask a possible action

            probs = self.brain_evl(obs)  # (1, 64)
            probs = probs.masked_fill(mask, -1e9)
            probs = torch.softmax(probs, dim=1)  # all masked prob equal to 0 after this step

            # e-greedy
            if np.random.uniform() < self.epsilon:
                action = torch.argmax(probs, dim=1).item()
                action = (action // 8, action % 8)
            else:
                # action = np.random.choice(range(64), p=probs.cpu().numpy().ravel())  # p-distribution
                action = random.choice(list(a_possible))
        return action

    def store_transition(self, obs, a, r, done, obs_, a_possible):
        """
        :param obs, obs_:list[list], 8x8
        :param a: tuple, (row, col)
        :param r: int
        :param done: bool
        :param a_candicates: a set of actions (row, col), it wont be used in the learning process so it does not matter if stored
        :return:
        """
        if self.id == "Blue":
            a_mask = np.ones(self.a_dim)
            for row, col in a_possible:
                a_mask[row * 8 + col] = 0  # do not mask a possible action
            transition = np.hstack((np.array(obs).ravel(), a[0]*8+a[1], r, done, np.array(obs_).ravel(), a_mask))  # (64+1+1+1+64+64)

            if self.prioritized:  # prioritized experience replay
                self.memory.store(transition)
            else:
                index = self.memory_counter % self.memory_size
                self.memory[index] = transition
                self.memory_counter += 1
                if self.memory_counter == self.memory_size*3:  # avoid too large number
                    self.memory_counter -= self.memory_size

    def weights_assign(self, another: Brain):
        """
        accept weights of the brain_eval from the blue player
        :param another:
        :return:
        """
        if self.id == "Red":
            with torch.no_grad():
                for tgt_param, src_param in zip(self.brain_evl.parameters(), another.parameters()):
                    tgt_param.data.copy_(self.alpha1 * src_param.data + (1.0 - self.alpha1) * tgt_param.data)

    def __tgt_evl_sync(self):
        """
        this function is to assign the weight of the eval network to the target network
        :return:
        """
        if self.id == "Blue":
            for tgt_param, src_param in zip(self.brain_tgt.parameters(), self.brain_evl.parameters()):
                tgt_param.data.copy_(self.alpha2 * src_param.data + (1.0 - self.alpha2) * tgt_param.data)

    def learn(self):
        if self.id == "Blue":  # only blue player learns
            self.brain_evl.train()
            self.brain_tgt.eval()  # do not train it.
            self.opt.zero_grad()
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.__tgt_evl_sync()

            if self.prioritized:
                tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
                ISWeights = torch.tensor(ISWeights, dtype=torch.float).squeeze().to(self.device)
            else:
                if self.memory_counter > self.memory_size:
                    sample_index = np.random.choice(self.memory_size, size=self.batch_size)
                else:
                    sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
                batch_memory = self.memory[sample_index]

            obs = torch.tensor(batch_memory[:, :self.s_dim], dtype=torch.float).to(self.device)
            a = torch.tensor(batch_memory[:, self.s_dim], dtype=torch.long).to(self.device)
            r = torch.tensor(batch_memory[:, self.s_dim+1], dtype=torch.float).to(self.device)
            done = torch.tensor(batch_memory[:, self.s_dim+2], dtype=torch.bool).to(self.device)
            obs_ = torch.tensor(batch_memory[:, -self.s_dim-self.a_dim: -self.a_dim], dtype=torch.float).to(self.device)
            # a_possible = batch_memory[:, -self.a_dim:]  # a_possbile is not used

            q_eval = self.brain_evl(obs)  # tensor, shape = [bs, 64]
            q_eval_wrt_a = torch.gather(q_eval, dim=1, index=a.view(-1, 1)).squeeze()  # [bs, ]
            with torch.no_grad():  # refer to https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_grad_enabled
                q_next = self.brain_tgt(obs_)  # tensor, shape = [bs, 64]
                
                # DQN
                q_target = r + self.gamma * torch.max(q_next, dim=1)[0]  # [bs, ]

                q_target[done] = r[done]  # [bs, ]

            if self.prioritized:
                with torch.no_grad():
                    abs_errors = torch.abs(q_target - q_eval_wrt_a).cpu().data.numpy()
                loss = torch.mean(ISWeights * torch.square(q_target - q_eval_wrt_a))
                self.memory.batch_update(tree_idx, abs_errors)
            else:
                loss = self.critic(q_target, q_eval_wrt_a)
                # loss = torch.mean(torch.square(q_target - q_eval_wrt_a))
            loss.backward()
            self.opt.step()
            self.learn_step_counter += 1
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            return loss.item()

    def reward_transition_update(self, reward:float):
        """
        if it is the red that take the last turn, the reward the blue player obtained should be updated because the winner has been determined
        :param reward: float
        :return:
        """
        if self.id == "Blue":
            if self.prioritized:
                index = (self.memory.tree.data_pointer - 1) % self.memory_size
                self.memory.tree.data[index][self.s_dim+1] = reward
            else:
                index = (self.memory_counter - 1) % self.memory_size
                self.memory[index, self.s_dim+1] = reward

    def save_model(self, name:str):
        torch.save(self.brain_evl.state_dict(), os.path.join("models", name))

    def load_model(self, name="brain_DQN"):
        if not os.path.exists(os.path.join("models", name)):
            sys.exit("cannot load %s" % name)
        self.brain_evl.load_state_dict(torch.load(os.path.join("models", name)))
