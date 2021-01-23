# Multi Discrete : 2021.01.24

여기서는 multidiscrete의 다양한 구현방법에 대해 직접 구현해보고 생각해본다.  
메인 목적이라고 한다면 방법들마다의 차이점을 정확히 아는것이지만 쉽지 않을듯하다.

Action Space Shaping in Deep Reinforcement Learning  
https://arxiv.org/pdf/2004.00980.pdf  
일단 이 논문을 정독하면 매우 도움이 될 것 같다. 각종 대회, 구현(OpenAI Five)에서 action space shaping을 어떻게 했고 성능을 어떻게 냈는가? 에 대한 서베이에 가까운 논문같다.  
https://github.com/Miffyli/rl-action-space-shaping  
심지어 코드도 있다.

일단 multidiscrete는 간단하게 예를들면 gridworld 환경에서 [좌, 우], [상, 하]로 action space가 2, 2인것이다.  
이 경우 대각선 이동이 된다. 총 액션의 수는 좌상, 좌하, 우상, 우하 이렇게 4개라고 볼 수 있다.  
다른 정의를 한다면 [noop, 좌, 우], [noop, 상, 하] 이런식으로 정의하게 되면 우리가 아는 자연스러운 8방향 이동이 나오게된다. 

이 파트의 가장 메인은 unity mlagents의 branch 시스템을 참고할 예정이다. 내가 본 시스템중 가장 잘 구현해놓은것 같다. 최근엔 아예 continuous가 0이여도 포함되도록 바뀌었더라. 통합하는건 좋은데 좀 이상하게 느껴지기도 한다.

방법을 분류해보자면

1. flatten
2. 완전 분리
3. 부분 액션
4. Multi-Discrete

이정도로 나눠볼 수 있을것같다.

여기선 환경은 MLAgent로 액션은 3, 3개로 FoodCollector 쓰던걸 가져와서 사용할 예정이다.
[noop, 전진, 후진], [noop, 좌회전, 우회전] 이렇게 조합된다.




## 위 논문에 대해서

일단 Table 1이 정말 유익한 자료이다. 결국 대회나 구현에서 시도했던것이 대부분 엄청난 대규모(OpenAI와 DeepMind)를 제외하면 Discrete쪽으로 최대한 reduce해서 시도했다는 이야기이다. 

실험이 아주 재미있는데, Tank라는 환경 세팅이 내가 시도하는 환경과 아주 유사하다. 사실 나의 목적은 MultiDiscrete 방법들간의 차이를 원하는거지만 이 논문에서는 여러가지 경우에 대해서 알아보는게 재미있다.

RA, DC, CMD라는게 나오는데, RA는 discrete에서 쓸모없는 액션을 지우는것, DC는 continuous를 이산화하는것, CMD는 multidiscrete를 flatten화 시키는것을 말한다.

여기서 backward와 strafe라는 말이 나오는데, backward는 후진, strafe는 좌우로 게걸음을 걷는것을 말한다. 

사실 환경에 따라서 backward나 strafe가 좋을수도 안좋을수도 있다. Get-To-Goal이 무슨 환경인지는 잘 모르겠는데, backward를 빼면 성능이 내려간다. 반면 Obstacle challenge에서는 성능이 올라간다. 이건 환경에 따라 케바케가 꽤 있을것같다.

일반적으로 discrete가 multidiscrete보다 성능이 좋게 나오는것을 볼 수 있는데... 이유가 뭘까? additional action에 robust하다고 하긴 한다.

