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
        self.fc_pi1 = nn.Linear(256, 3)
        self.fc_pi2 = nn.Linear(256, 3)
        self.fc_v  = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        out1 = F.relu(self.fc1(x))
        out_pi1 = self.fc_pi1(out1)
        out_pi2 = self.fc_pi2(out1)
        prob1 = F.softmax(out_pi1, dim=softmax_dim)
        prob2 = F.softmax(out_pi2, dim=softmax_dim)
        return prob1, prob2
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a1_lst, a2_lst, r_lst, s_prime_lst, prob1_a_lst, prob2_a_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a1, a2, r, s_prime, prob1_a, prob2_a, done = transition
            
            s_lst.append(s)
            a1_lst.append([a1])
            a2_lst.append([a2])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob1_a_lst.append([prob1_a])
            prob2_a_lst.append([prob2_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s, a1, a2, r, s_prime, done_mask, prob1_a, prob2_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a1_lst), torch.tensor(a2_lst), \
                                                             torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                             torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob1_a_lst), torch.tensor(prob2_a_lst)
        self.data = []
        return s, a1, a2, r, s_prime, done_mask, prob1_a, prob2_a
        
    def train_net(self):
        s, a1, a2, r, s_prime, done_mask, prob1_a, prob2_a = self.make_batch()
        
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

            pi1, pi2 = self.pi(s, softmax_dim=1)

            pi1_a = pi1.gather(1, a1)
            pi2_a = pi2.gather(1, a2)

            pi_a = torch.cat((pi1_a, pi2_a), 1)
            prob_a = torch.cat((prob1_a, prob2_a), 1)

            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            
            pi_loss = -torch.mean(torch.min(surr1, surr2))
            v_loss = F.smooth_l1_loss(self.v(s), td_target.detach())
         
            loss = pi_loss + v_loss

            self.optimizer.zero_grad()
            loss.backward()
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
                prob1, prob2 = model.pi(torch.from_numpy(s).float())
                m1 = Categorical(prob1)
                m2 = Categorical(prob2)
                a1 = m1.sample().item()
                a2 = m2.sample().item()
                
                s_prime, r, done, info = env.step(np.array([a1, a2]))

                model.put_data((s, a1, a2, r, s_prime, prob1[a1].item(), prob2[a2].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        # if n_epi % 50 == 0 and n_epi != 0:
        #     torch.save(model.state_dict(), f"ppo_model_{n_epi}.pt")
        #     print("saved!")
        #     break

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()


if __name__ == "__main__":
    train()
