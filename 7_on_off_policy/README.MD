# on policy / off policy

## on policy / off policy의 차이

난 강화학습 공부를 시작하고나서 아직까지도 가장 아리까리한 부분이 이 개념이다. 종잡을 수 없는 개념이다. 누가 확 잡아주면 좋을것같다.

### 정의
behavior policy와 target policy가 같으면 on-policy, 다르면 off-policy라고 함

## 기존의 이야기들  
DQN은 off-policy, SARSA는 on-policy이다. 일반적으로 이렇게 알려져있다. 내가 이해한바로는 value 기반에서는 대부분 maxQ를 사용하니까 off-policy이다. 라는 논리라고 이해가 되었다.

잠깐 maxQ의 장점에 대해 살펴보자. 이 방법은 어찌보면 참 궁극의 방법이기도 한데 각 state에서 가장 좋은 행동을 배우자는 아주 greedy한 방법이다. 이 Q값은 이렇게 저렇게 바뀌어도 결국 maxQ라는게 변하면서도 존재하는 어떤 타겟이 되어준다.

policy는 학습이 끝나고 나면 일반적으로는 좋은 행동이 높은 확률을 갖지만 도중에는 사실 어느 policy가 우월한지 알 수 없다. 이 점이 maxQ와의 가장 큰 차이라고 볼 수 있다. Q값은 그 자체만으로도 현재의 optimal을 따라가는데에 무리가 없다. 결국 value network를 통해 평가를 진행해야한다. 

OpenAI spinningup에도 value 기반 기법은 전부 off policy라고 나와있다. maxQ를 써야하니까. 반면 policy 기법들은 on policy라고 나와있다. 

여기서 ppo가 on-policy라고 분류되는 이유가 매우 재미있으면서도 어렵다. ppo는 샘플링한 traj를 k epoch만큼 학습시킨다. 그렇다면 최초의 policy와 최종 직전의 policy는 같은 policy인가? 학습식을 보면 실제로 target policy가 계속 바뀌어나간다. 이건 off policy인거같은데? 근데 신뢰구간 안이니까 괜찮다. 라고 말하는거같다. 

스스로 off policy라고 말하는 부분에서도 재미있는것도 있다.
```
However, because the policy used to generate a trajectory can lag behind the policy on the learner by several updates at the time of gradient calculation, learning becomes off-policy. - 출처 IMPALA 논문
```
위에서 말한것처럼, 많은 에이전트들이 샘플링하고 그 데이터로 학습하게되면 actor와 learner는 벌어지게 된다. 여기서 actor가 behavior policy이고, learner는 target policy라고 볼 수 있다. 임팔라의 특성상 괴리가 생길수밖에 없다. 하지만 그 괴리가 ppo에서 k값을 크게 했을때보다 클까? 난 이 부분이 매우 궁금했다. 

spinning up의 ppo 구현을 보면 무려 80번이나 반복한다. openai baselines는 4번 한다. 다른 구현들도 살펴보면 일반적으로 k는 10번 이하의 작은 숫자이긴 하다. 

## 이후의 생각

