# cartpole example

import gym
import torch

env = gym.make('CartPole-v1')

# init simple neural net
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(4, 100)
        self.fc2 = torch.nn.Linear(100, 2)
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


best_steps = -1

# implement bayesian optimization


for epoch in range(1000):

    # reinit net
    # net = Net()
    # load model
    net = Net()
    # net.load_state_dict(torch.load("best.pth"))

    end_steps = []
    for i_episode in range(20):
        observation, info = env.reset()
        for t in range(1000):
            env.render()
            #print(observation)
            out = net(torch.tensor(observation, dtype=torch.float32))

            # get action from out
            action = torch.argmax(out).item()
            # action = env.action_space.sample()

            observation, reward, done, turncated, info = env.step(action)
            #print(observation, reward, done, info)
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
        end_steps.append(t)
    
    print("Epoch: {}, Avg. steps: {}".format(epoch, sum(end_steps)/len(end_steps)))

    if sum(end_steps)/len(end_steps) > best_steps:
        best_steps = sum(end_steps)/len(end_steps)
        torch.save(net.state_dict(), "best.pth")
        print("New best model saved!")