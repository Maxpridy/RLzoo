import torch
import torch.nn as nn
import torch.distributions
import torch.nn.functional as F

import numpy as np

class Test1(nn.Module):
    def __init__(self):
        super(Test1, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        output = self.output(out2)
        return output


torch.manual_seed(42)
np.random.seed(42)

model = Test1()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for _ in range(10):
    input = np.array([[1 for _ in range(2)], [1 for _ in range(2)]])

    pred = model(torch.from_numpy(input).float())

    loss = torch.sum((pred - 1)**2)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
torch.save(model.state_dict(), "model.pt")