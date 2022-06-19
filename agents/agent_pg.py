import os
import sys
import torch
import torch.nn as nn
import numpy as np

from agents.utils import Brain
np.random.seed(47)

class Agent_PG(nn.Module):
    def __init__(self, player="Red", device=torch.device("cuda")):
        super(Agent_PG, self).__init__()
        assert player in ("Red", "Blue"), "Wrong player color"
        self.id = player
        self.a_dim = 64
        self.s_dim = 64
        self.device = device
        self.brain = Brain(a_dim=self.a_dim, s_dim=self.s_dim).to(device)
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_a_cand = [], [], [], []
        self.gamma = 0.95  # reward decay rate
        self.alpha = 0.2  # soft copy weights from blue to red, alpha updates while (1-alpha) remains
        if self.id == "Blue":
            self.opt = torch.optim.Adam(self.brain.parameters(), lr=1e-4, weight_decay=0.01)
            self.critic = nn.CrossEntropyLoss(reduction="none")  # do not apply mean

    def choose_action(self, obs, a_candicates):
        """
        :param obs: list[list], [8, 8]
        :param a_candicates: a set of tuples
        :return: a tuple of (row, col)
        """
        self.brain.eval()
        obs = torch.tensor(np.array(obs).ravel(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mask = torch.tensor([[True]*64], dtype=torch.bool).to(self.device)  # shape = [1, 64]
            for r, c in a_candicates:
                mask[0][r * 8 + c] = False   # do not mask a possible action

            probs = self.brain(obs)
            probs = probs.masked_fill(mask, -1e9)
            probs = torch.softmax(probs, dim=1)

            # action = torch.argmax(probs, dim=1).item()  # greedy
            action = np.random.choice(range(64), p=probs.cpu().numpy().ravel())

            action = (action // 8, action % 8)
        return action

    def store_transition(self, obs, a, r, a_candicates):
        """
        :param obs:list[list], 8x8
        :param a: tuple, (row, col)
        :param r: int
        :param a_candicates: a set of actions (row, col)
        :return:
        """
        self.ep_obs.append(np.array(obs).ravel())
        self.ep_as.append(a[0] * 8 + a[1])
        self.ep_rs.append(r)
        self.ep_a_cand.append(a_candicates)

    def learn(self):
        if self.id == "Blue":
            # discount and normalize episode reward
            discounted_ep_rs_norm = torch.tensor(self.__discount_and_norm_rewards()).to(self.device)
            self.brain.train()
            self.opt.zero_grad()
            obs = torch.tensor(np.vstack(self.ep_obs), dtype=torch.float32).to(self.device)
            labels = torch.tensor(np.array(self.ep_as), dtype=torch.long).to(self.device)
            masks = torch.tensor([[True]*64 for _ in range(len(self.ep_a_cand))]).to(self.device)
            for i, a_cand in enumerate(self.ep_a_cand):
                for r, c in a_cand:
                    masks[i][r * 8 + c] = False

            net_out = self.brain(obs)
            net_out = net_out.masked_fill(masks, -1e9)

            # log_softmax = nn.LogSoftmax(dim=1)(net_out)
            # neg_log_likelihood = nn.NLLLoss(reduction="none")(log_softmax, labels)
            # loss = torch.mul(neg_log_likelihood, discounted_ep_rs_norm)
            # loss = loss.mean()
            loss = torch.mean(torch.mul(self.critic(net_out, labels), discounted_ep_rs_norm))

            loss.backward()
            self.opt.step()
            self.ep_obs, self.ep_as, self.ep_rs, self.ep_a_cand = [], [], [], []
            return loss.item()

    def __discount_and_norm_rewards(self):
        # https://www.janisklaise.com/post/rl-policy-gradients/
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float64)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def weights_assign(self, another:Brain):
        for tgt_param, src_param in zip(self.brain.parameters(), another.parameters()):
            tgt_param.data.copy_(self.alpha * src_param.data + (1. - self.alpha) * tgt_param.data)

    def save_model(self, name:str):
        torch.save(self.brain.state_dict(), os.path.join("models", name))

    def load_model(self, name="brain_PG"):
        if not os.path.exists(os.path.join("models", name)):
            sys.exit("cannot load %s" % name)
        self.brain.load_state_dict(torch.load(os.path.join("models", name)))
