# 이곳에서 테스트할 내용
# 모델 크기 / 얼마나 자주 모델을 갈아주는지에 대한 빈도
# 두 주제를 중점으로 확인할 것


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, linear_size):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4, linear_size)
        self.fc_pi = nn.Linear(linear_size, 2)
        self.fc_v  = nn.Linear(linear_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, teacher_prob_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done, teacher_prob = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            teacher_prob_lst.append(teacher_prob)
            
        s = torch.tensor(s_lst, dtype=torch.float)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst)
        torch_teacher_prob = torch.tensor(teacher_prob_lst)
    
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, torch_teacher_prob
        
    def train_net(self, is_first):
        s, a, r, s_prime, done_mask, prob_a, teacher_prob = self.make_batch()

        if is_first:
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

                pi = self.pi(s, softmax_dim=1)
                pi_a = pi.gather(1,a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        else:
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

                pi = self.pi(s, softmax_dim=1)

                kl_lossfunc = torch.nn.KLDivLoss(reduction='batchmean')
                kl_loss = kl_lossfunc(pi.log(), teacher_prob)
                
                pi_a = pi.gather(1,a)

                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach()) + 0.001*kl_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
def main():
    env = gym.make('CartPole-v1')
    
    model_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    now_size = model_sizes[0]

    teacher_model = PPO(linear_size=now_size)
    student_model = PPO(linear_size=now_size)

    score = 0.0
    print_interval = 20

    change_interval = 100
    is_first = True

    for n_epi in range(10000):
        s = env.reset()
        done = False

        if n_epi % change_interval == 0 and n_epi != 0:
            print(f"change model! teacher_size:{now_size}, student_size:{now_size+10}")
            is_first = False
            # student를 저장
            torch.save(student_model.state_dict(), "student.pt")
            
            # teacher에 student를 불러옴
            teacher_model = PPO(linear_size=now_size)
            teacher_model.load_state_dict(torch.load("student.pt"))
            teacher_model.eval()

            # student init
            now_size = model_sizes[n_epi//change_interval]
            student_model = PPO(linear_size=now_size)


        while not done:
            for t in range(T_horizon):
                prob = student_model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
            
                if is_first:
                    student_model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done, [0, 0]))
                else:
                    teacher_prob = teacher_model.pi(torch.from_numpy(s).float())
                    student_model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done, [teacher_prob[0].item(), teacher_prob[1].item()]))
                s = s_prime

                score += r
                if done:
                    break

            student_model.train_net(is_first)

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()