import torch
import torch.nn as nn
import torch.distributions
import torch.nn.functional as F

import numpy as np

class Test1(nn.Module):
    def __init__(self):
        super(Test1, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.output1 = nn.Linear(64, 2)
        self.output2 = nn.Linear(64, 2)

    def forward(self, x, softmax_dim=-1):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        return F.softmax(self.output1(out2), dim=softmax_dim), F.softmax(self.output2(out2), dim=softmax_dim)

class Test2(nn.Module):
    def __init__(self):
        super(Test2, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.output1 = nn.Linear(64, 4)

    def forward(self, x, softmax_dim=-1):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        return F.softmax(self.output1(out2), dim=softmax_dim)


torch.manual_seed(42)
np.random.seed(42)

#model = Test1()
model = Test2()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()

print(optimizer.state_dict())


for _ in range(100):
    input = np.array([[1 for _ in range(11)], [2 for _ in range(11)], [3 for _ in range(11)], [4 for _ in range(11)]])
    pred = model(torch.from_numpy(input).float())

    print()
    print(pred)
    
    x = pred
    y = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float, requires_grad=False)
    # 이걸 2x2로 바꿔말하면, (0, 0), (0, 1), (1, 0), (1, 1)
   
    loss = 0
    for e, f in zip(y, x):
        loss += -1 * e * torch.log(f+1e-7) # - ylogx
    
    loss = torch.mean(loss)

    optimizer.zero_grad()
    loss.backward()

    #print(model.last[0].weight.grad)
    
    optimizer.step()