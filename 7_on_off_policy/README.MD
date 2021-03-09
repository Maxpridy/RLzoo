# on policy / off policy

## on policy / off policy의 차이

난 강화학습 공부를 시작하고나서 아직까지도 투탑으로 아리까리한 부분이 이 개념이다. 종잡을 수 없는 개념이다. 누가 확 잡아주면 좋을것같다.

### 정의
behavior policy와 target policy가 같으면 on-policy, 다르면 off-policy라고 함

## 기존의 이야기들  
DQN은 off-policy, SARSA는 on-policy이다. 일반적으로 이렇게 알려져있다. 내가 이해한바로는 value 기반에서는 학습된 policy가 (현재의)최적의 정책이자 value function을 포함하기 때문에, 그리고 exploration을 해야하기 때문에 off-policy이다. 라는 논리로 이해하고 있다.

```
We’ve mentioned that DDPG is an off-policy algorithm: this is as good a point as any to highlight why and how. Observe that the replay buffer should contain old experiences, even though they might have been obtained using an outdated policy. Why are we able to use these at all? The reason is that the Bellman equation doesn’t care which transition tuples are used, or how the actions were selected, or what happens after a given transition, because the optimal Q-function should satisfy the Bellman equation for all possible transitions. So any transitions that we’ve ever experienced are fair game when trying to fit a Q-function approximator via MSBE minimization.
```
-출처 openai spinningup DDPG 설명 중

먼저 DDPG의 behavior policy를 보자. mu(s)+noise()가 액션이 된다. 하지만 target policy에선 노이즈가 없다. 왜 다른가? exploration을 해야하니까.  
위 설명은 왜 off policy로 작동해도 되는가? 에 대한 설명이다. DDPG는 continuous action이기 때문에 discrete action space DQN과는 조금 다른 경우라고 볼 수 있다.

여기까지 오면 off policy가 무엇인지 이제는 알 수 있다. value 기반의 방법들은 behavior policy에서는 exploration을 포함하고, target policy에서는 순수한 policy의 아웃풋을 추정해 나가면서 학습이 되는것이다. 이것은 Q와 SARSA에서도 마찬가지로 비교해볼 수 있다.

우리가 일반적으로 보아왔듯이 transition에 대한 데이터(s, a, r, s', d)를 쌓아나감으로써 우린 그 데이터를 이후에 학습할 수 있게된다. 왜 학습이 가능한가? 위 DDPG에서 설명한것처럼 
1. 그 튜플이 사용되었는지 신경쓰지 않고
2. 어떤 액션이 선택되었는지도 신경쓰지 않고
3. transition 후에 일어나는 일도 신경쓰지 않는다.  

왜냐면 우리가 목표로 하는건 벨만 식을 만족시키는 Q값을 학습시키는것이고, 저 데이터만으로도 식을 만족시킬 수 있기 때문이다. 아래 식을 보면 우리가 과거 데이터를 사용한다고 해도 문제가 될것은 없다. 거기에 maxQ와 deterministicQ는 현재로써 최고의 목표를 지정해준다.
```
# DQN의 타겟식
target = r + gamma * q_target(s_prime).max(1)[0].unsqueeze(1) * done_mask
```
```
# DDPG의 타겟식
target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
```
지금까지 말한것은 사실 off policy니까 experience replay기법을 쓸 수 있다는것처럼 들리지만 사실 value기반이니까 쓸 수 있다가 맞다고 본다.

그럼 policy 기반은 왜 일반적으로 불가능한가? 

A2C가 학습되는 방식을 보자.
```
# value가 학습되는건 비슷하니까 제외함
loss = -(torch.log(pi_a) * advantage.detach()).mean()
```
한참 학습이 되었고, 데이터도 모았다고 가정해보자. 위 식을 과거의 학습되던 (s, a, r, s', d)같은 튜플을 이용해서 학습하려고 해보자. 가능한가? 여기서 우리가 목표로 하는것은 return을 최대로하는 policy를 학습시키는것이다. 위와는 다르다. pi_a를 구하기 위해서 pi(s)를 넣는 순간 이 pi의 출력은 저 튜플을 샘플할때의 pi와 같을까? 그럴리가 없다. 그럼 이 샘플의 데이터들과 현재 pi로 나오는 데이터들의 출력은 당연히 달라질 수 밖에 없다. 그럼 나는 저장한 데이터를 활용하려면 어떻게 해야할까? true value를 구하기 위한 계산을 해야할것이다. 이렇기 때문에 과거 데이터를 활용하려면 IS같은 특이한 방법을 사용해야하는 상황이 되는것이다. (ACER / IMPALA 등) 

OpenAI spinningup에도 value 기반 기법은 전부 off policy라고 나와있다. 반면 policy 기법들은 on policy라고 나와있다. 

여기서 ppo가 on-policy라고 분류되는 이유가 매우 재미있으면서도 어렵다. ppo는 샘플링한 traj를 k epoch만큼 학습시킨다. 그렇다면 최초의 policy와 최종 epoch 직전의 policy는 같은 policy인가? 학습식을 보면 실제로 target policy가 계속 바뀌어나간다. 이건 off policy인거같은데? 근데 신뢰구간 안이니까 괜찮다. 라고 말하는거같다. 

스스로 off policy라고 말하는 부분에서도 재미있는것도 있다.
```
However, because the policy used to generate a trajectory can lag behind the policy on the learner by several updates at the time of gradient calculation, learning becomes off-policy. 
```
-출처 IMPALA 논문  

위에서 말한것처럼, 많은 에이전트들이 샘플링하고 그 데이터로 학습하게되면 actor와 learner는 벌어지게 된다. 여기서 actor가 behavior policy이고, learner는 target policy라고 볼 수 있다. 임팔라의 특성상 괴리가 생길수밖에 없다. 하지만 그 괴리가 ppo에서 k값을 크게 했을때보다 클까? 난 이 부분이 매우 궁금했다. 

spinning up의 ppo 구현을 보면 무려 80번이나 반복한다. openai baselines는 4번 한다. 다른 구현들도 살펴보면 일반적으로 k는 10번 이하의 작은 숫자이긴 하다. 

## 실험 진행 예정



## 이후의 생각

