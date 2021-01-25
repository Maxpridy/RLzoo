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


# torch.manual_seed(42)
# np.random.seed(42)

model = Test1()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()

print(optimizer.state_dict())

for _ in range(100):
    input = np.array([[1 for _ in range(11)], [2 for _ in range(11)], [3 for _ in range(11)], [4 for _ in range(11)]])
    pred1, pred2 = model(torch.from_numpy(input).float())

    print()
    print(pred1)
    print(pred2)
    
    x = pred1
    y = torch.tensor([[[1, 0], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, 1], [0, 1]]], dtype=torch.float, requires_grad=False)
    
    loss = 0
    
    for e, f in zip(y, pred1):
        loss += -1 * e[0] * torch.log(f+1e-7) # - ylogx

    for e, f in zip(y, pred2):
        loss += -1 * e[1] * torch.log(f+1e-7) # - ylogx

    # 여기서 동기인 신모씨가 도움을 좀 주었는데, 데이터와 라벨의 상관관계에 대해 파악할수록 좋다는 말을 해주었다. 굉장히 좋은말인것같다.
    # 예를들면 이 경우에선 test1의 데이터와 0, 1, 2, 3 라벨은 1:1이라고 관계라도 해도 될정도로 서로에게 간섭을 하지 않는다.
    # 반대로 test2에서는 [1, ...], [2, ...], [3, ...], [4, ...]라는 데이터를 만약 multi class로 라벨링하면 1, 2, 3, 4 각각은 모델 입장에서는 어떤 관계를 갖는다. 둘씩 뽑아보면 부분적으로 같은 출력을 내기 때문이다.
    
    # 신모씨가 시도해본 경우에 대해 소개해주었는데, 카카오 아레나 쇼핑카테고리분류를 예로 든다면 모든 클래스를 단순히 곱하면 40549544640개가 나온다. 
    # 하지만 train에 존재하는 클래스 집합관계들만 라벨로 사용한다면 그 경우는 매우 크게 감소한다. 실제로는 데이터에 컴퓨터 하위에 옷이 존재하거나 그 반대는 없기 때문이다.
    # 따라서 class를 잘 골라내면 flatten이 불리할 이유는 없을것이다. 애초에 대규모 라벨링에서 모든 관계가 있는 라벨이라는게 존재는 할 수 있을까? 소규모(ex, 개,고양이/단모,장모)같은 경우에선 모든 관계의 라벨이 말이 되긴 하지만..

    # pred1이 먼저 학습되는 이유?
    # ?? 모르겠음

    loss = torch.mean(loss)

    optimizer.zero_grad()
    loss.backward()

    #print(model.last[0].weight.grad)
    
    optimizer.step()