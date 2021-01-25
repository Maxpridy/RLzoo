# flatten

branch 정보를 받아서 flatten화 한다.

사실 표현이 완전히 정립된건 아닌것 같고 여러가지 표현이 있는데 MultiDiscrete를 일자로 만드는건 flatten이라고 하는것같고, reduced action space라고도 부르기도 한다. 내가 생각하기엔 reduced action space는 논문에서 말하는 CMD와 RA를 둘다 포함하는 개념이라고 생각된다.

논문에서는 CMD라고 표현하더라. Convert multi-discrete actions to discrete

여기서 사용한 예시는 3x3인데, [noop, 전진, 후진], [noop, 좌회전, 우회전]으로 총 액션의 수는 9가 된다.  
이 액션들을 나열하면 [(noop, noop), (noop, 좌회전), (noop, 우회전), (전진, noop), (전진, 좌회전), (전진, 우회전), (후진, noop), (후진, 좌회전), (후진, 우회전)] 이렇게 된다.

flatten의 장점은 쓸데없는 액션을 masking하기가 굉장히 용이하다는것이다. 예를들면 내가 학습시키고자 하는 agent의 행동이 후진이 필요가 없다고 생각해보자. 그러면 0~8 인덱스중 6, 7, 8번 인덱스 액션은 필요가 없어진다. 가만히 있는것을 허용하지 않으려면 0, 1, 2번도 지울 수 있다. 그러면 policy의 출력이 3개인 모델을 학습시키면 되는것이다. 특별한 경우에 더 쓸모있는데, 딱 0번 인덱스만 없애면 쉽게 noop를 하지 않는 action space를 만들 수 있다.

단점은 전체 action space가 모두 필요할 때 총 액션의 수가 기하급수적으로 증가하게 된다. branch가 10개, 브랜치마다 3개의 액션이라고 한다면 3^10이 총 액션이 갯수가 된다. 이게 사실 단점인지는 잘 모르겠다. multidiscrete라고 상황이 다른건 아니니까. 다만 액션의 표현이 숫자 하나인 discrete에 비해 multidiscrete쪽이 표현하기 좋다는점은 확실하다.


