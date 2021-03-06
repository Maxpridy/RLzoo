# 문도 피구

1. 환경 설명  
유저들이 만들어낸 리그오브레전드 내의 미니게임. 두 플레이어는 룰에 따라 바론둥지에 모여 게임을 시작한다. 게임 방법은 두 문도가 서로에게 식칼을 던지는것이고 룰은 다양하기 때문에 적당히 적용하면 된다.

2. 환경 구축  
두가지 큰 고민이 있었는데 하나는 난이도에 대한 고민이였고, 다른 하나는 잘해보이기 위해선 어떻게 해야하는가에 대한 고민이였다.  
환경의 목적은 '인간이 보기에 잘해보이는것'을 목적으로 했다. 따라서 문도 피구의 최적 전략중 하나인 다가가서 먼저 맞추고나서 피할 생각도 없이 쿨타임마다 무작정 던지는 전략(그냥 학습시키면 이런 전략을 배운다)을 학습시키는것을 최대한 지양했다. 잘해보이기 위해선 이런 전략이 아닌 상대방의 식칼은 피하고 내 식칼은 모두 맞추는것이 사람이 보기에 잘하는 에이전트라고 판단했고 그쪽으로 학습시키려고 했다.  
환경의 구체적인 상황은 왼쪽은 딥러닝 에이전트, 오른쪽은 룰베이스의 에이전트로 기획했다. 처음에는 순수 강화학습으로 양쪽을 번갈아가면서 학습시킬 생각이였으나 생각보다 피하고 - 맞추고 - 피하고 - 맞추는것을 반복하는것을 학습시키는것은 쉽지 않았다. 그래서 최종적으로는 위에 말한것과 같이 좌측에만 딥러닝 에이전트가 존재하도록 설계했다.  
오른쪽에 있는 룰베이스 에이전트는 간단한 과정을 반복한다. [다가가기 -> 던지기 -> 물러나기 -> 도중에 식칼이 날아오면 직각으로 피하기] 이런 과정을 거친다. 여기서 던지기에 대해 양쪽 에이전트의 행동에 대해 고민을 했는데, 딥러닝 에이전트(좌측)은 정확히 던지게 했고, 룰베이스 에이전트(우측)은 약간의 랜덤한 각도를 포함해서 던지게 했다. 이것은 문제를 약간 쉽게 만들기 위함이였다. 룰베이스 에이전트는 식칼 오브젝트가 생성되자마자 직각무빙을 하기때문에 거리가 어느정도 벌어져있으면 무조건 피할 수 있기때문에 딥러닝 에이전트가 정확한 위치에 던지도록 설정했다. 그 반대로 딥러닝 에이전트는 상대방이 어떤 랜덤 각도로 던지느냐에 따라 어떻게든 직각 비슷한 방향으로 움직이기만 하면 피할 가능성이 생기게 된다. 만약 정확히 던진다면 일정 거리 이상 벌어지지 않는다면 무조건 맞을것이다.  
환경을 설계하면서 다시한번 깨달았는데 문도피구만이 아닌 롤에서 논타겟팅은 100% 맞출 수 있는 상황이 굉장히 많다. 단지 사람이 못맞추는것일 뿐이다. 이것때문에 환경에 대한 밸런싱을 약간 더 고민한 것 같다.  
state는 ml-agents에서 제공하는 ray perception sensor 3d와 몇가지 위치정보(내 위치, 상대방의 위치, 상대방 식칼의 존재여부 등)를 사용했다. 비전 정보는 사용하지 않았고 CNN도 사용하지 않았다.  
reward는 중간에 reward shaping을 몇번 하긴 했지만 최대한 직관적으로 dense하게 주었다. 예를들면 내 식칼이 상대방을 맞추면 소모된 상대방의 hp에 비례하도록, 내가 상대방의 식칼을 피해서 상대방의 문도가 체력이 달면(50 소모) 그정도의 reward를 주는식으로 설계했다.  
action은 multidiscrete로 branch가 3, branch마다 3, 3, 2의 discrete action space를 사용했다. 앞의 3, 3은 noop, 상, 하 / noop, 좌, 우 이고 2는 noop / 던지기 로 설정했다.  

3. 실험 방법  
Unity로 게임을 빌드하여 ML-Agents executable로 불러와 학습시켰다. SL+RL을 사용했고 SL에 사용한 데이터는 50게임정도 직접 플레이해서 생성했다. RL은 PPO를 사용했다.  
실험영상 : https://www.youtube.com/watch?v=bVW6-CJSQWU

4. 실험 결과  
인간이 보기에 잘해보이는것을 보다 구체적으로 말하면 '가까이가서 던지고, 멀리 가서 피한다'라고도 볼 수 있다. 이 전략은 승리의 가능성을 높이는 전략이면서도 사람이 보기에 영리해 보이는 전략이기 때문이다. 내가 직접 SL 데이터를 만들면서 플레이 했을때에도 오른쪽의 룰베이스 에이전트에게 가끔 지기도 했는데 가끔 이기는것만으로도 학습이 되었다고 할만하다고 생각한다.
