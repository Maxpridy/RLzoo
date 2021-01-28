import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

import my_env

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(77, 256)
        self.fc_pi_a = nn.Linear(256, 3)
        self.fc_pi_b1 = nn.Linear(256, 3)
        self.fc_pi_b2 = nn.Linear(256, 3)
        self.fc_pi_b3 = nn.Linear(256, 3)
        self.fc_v  = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        out1 = F.relu(self.fc1(x))
        pi_a = self.fc_pi_a(out1)
        pi_b1 = self.fc_pi_b1(out1)
        pi_b2 = self.fc_pi_b2(out1)
        pi_b3 = self.fc_pi_b3(out1)
        return F.softmax(pi_a, dim=softmax_dim), F.softmax(pi_b1, dim=softmax_dim), F.softmax(pi_b2, dim=softmax_dim), F.softmax(pi_b3, dim=softmax_dim)
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, b_lst, r_lst, s_prime_lst, prob_a_lst, eye_list, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, b, r, s_prime, prob_a, eyes, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            b_lst.append([b])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            eye_list.append(eyes)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s, a, b, r, s_prime, done_mask, prob_a, eye = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(b_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), torch.tensor(eye_list)
        self.data = []
        return s, a, b, r, s_prime, done_mask, prob_a, eye
        
    def train_net(self):
        s, a, b, r, s_prime, done_mask, prob_a_b, eye = self.make_batch()
        
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi_a, pi_b1, pi_b2, pi_b3 = self.pi(s, softmax_dim=1)

            mixed_b = torch.unsqueeze(eye[:, 0], 1)*pi_b1 + torch.unsqueeze(eye[:, 1], 1)*pi_b2 + torch.unsqueeze(eye[:, 2], 1)*pi_b3
            
            gather_pi_a = pi_a.gather(1, a)
            gather_pi_b = mixed_b.gather(1, b)
            
            multiply_a_b = gather_pi_a*gather_pi_b

            ratio = torch.exp(torch.log(multiply_a_b) - torch.log(prob_a_b))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def train():
    env = my_env.MyEnv(0)
    model = PPO()
    score = 0.0
    print_interval = 1

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                a_prob, pi_b1, pi_b2, pi_b3 = model.pi(torch.from_numpy(s).float())
                a_m = Categorical(a_prob)
                a = a_m.sample().item()
                if a == 0:
                    b1_m = Categorical(pi_b1)
                    b1 = b1_m.sample().item()
                    b = b1
                    a_b_prob = a_prob[a] * pi_b1[b1]
                elif a == 1:
                    b2_m = Categorical(pi_b2)
                    b2 = b2_m.sample().item()
                    b = b2
                    a_b_prob = a_prob[a] * pi_b2[b2]
                else:
                    b3_m = Categorical(pi_b3)
                    b3 = b3_m.sample().item()
                    b = b3
                    a_b_prob = a_prob[a] * pi_b3[b3]

                action = a*3 + b

                s_prime, r, done, info = env.step(action)

                model.put_data((s, a, b, r, s_prime, a_b_prob, np.eye(3)[a], done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % 50 == 0 and n_epi != 0:
            torch.save(model.state_dict(), f"ppo_model_{n_epi}.pt")
            print("saved!")
            break

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()


if __name__ == "__main__":
    train()
