import torch
import torch.nn as nn
import torch.distributions
import torch.nn.functional as F

import numpy as np

class Test1(nn.Module):
    def __init__(self):
        super(Test1, self).__init__()
        self.last = nn.Sequential(
            nn.Linear(11, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x, softmax_dim=-1):
        x = self.last(x)
        return F.softmax(x, dim=softmax_dim)


model = Test1()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()

print(optimizer.state_dict())

for _ in range(1000):
    input = np.array([[1 for _ in range(11)], [1 for _ in range(11)]])
    probs = model(torch.from_numpy(input).float())

    print()
    m = torch.distributions.Categorical(probs)
    
    x = probs
    y = torch.tensor([[0, 0, 0.8, 0, 0.2, 0], [0, 0, 0.75, 0, 0.15, 0]], dtype=torch.float, requires_grad=False)
   
    log_x = torch.log(x+1e-6)
    log_y = torch.log(y+1e-6)

    # forward
    kl_1 = torch.mean(torch.sum(torch.exp(log_y) * (log_y - log_x), dim=-1)) # ylogy - ylogx
    
    # reverse
    kl_2 = torch.mean(torch.sum(torch.exp(log_x) * (log_x - log_y), dim=-1)) # xlogx - xlogy

    # CE
    loss1 = 0
    for e, f in zip(y, x):
        loss1 += -1 * e * torch.log(f+1e-7) # - ylogx

    # reverse CE
    loss2 = 0
    for e, f in zip(x, y):
        loss2 += -1 * e * torch.log(f+1e-7) # - xlogy

    #loss = kl_1
    loss = torch.sum(loss2)
    # soft 라벨에 대해선 reverse CE만 작동하지 않음. -xlogy만으론 안됨. xlogx - xlogy여야 학습이 가능하고 kl_1과 같은곳으로 수렴한다고함. 
    # 간단하게 이해하면 x의 엔트로피를 최소화하는 의미라고 이해되는데, 정확히는 잘 모르겠음

    print(probs)
    
    optimizer.zero_grad()  
    loss.backward()
    optimizer.step()