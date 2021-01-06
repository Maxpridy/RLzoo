# 4. valueDICE 적용해보기

https://arxiv.org/pdf/1912.05032v1.pdf

valueDICE 논문은 이쪽이고 논문에 구글리서치 코드도 공개되어있다.  
내용적인 면은 살펴보면서 알아보자


## 테스트해볼 환경 탐색

원래는 NLE로 해볼 생각이였는데 NLE는 state가 꽤 복잡하기도 하고 데이터 얻는법이 명확하지가 않은 느낌이였다. ttyrec으로 게임 화면 자체는 저장하긴 하는거같은데...  사실 가장 중요한 문제라면 내가 넷핵을 잘하지 못한다.

다른 환경들도 대부분 imitation 테스트하기에 썩 좋지는 않다. 그래서 직접 만들려고 한다. Unity ML-agents의 GridWorld를 복제해서 비슷한 환경을 만들어보려고 한다.


## 환경 기획

뭐니뭐니해도 imitation이 많이 필요한 환경은 reward가 굉장히 sparse할 때이다. 리워드가 필요 없는 imitation도 많이 있기때문에(valueDICE도 그런듯함) 리워드는 있는둥 없는둥 해도 괜찮을것 같다.

그냥 GridWorld를 똑같이 해도 될것같은데 잘 모르겠다. 오히려 일반화 성능도 검증할 수 있으니 좋다고 봐야할지... 원래 5x5가 기본값인데 일단 테스트를 진행해보면 어떨까싶다. 커질수록 기하급수적으로 어려워져서 (특히 일반화가) 난이도에 대한 감을 잡는게 참 어려운것같다.


## 에이전트 기획

논문에서는 mujoco에 continuous 환경들을 대상으로 실험이 되어있다. continuous control 물론 좋지만 사실 가장 보편적것이 discrete 환경이기 때문에 말했듯이 GridWorld에서 discrete action space에 대한 imitation을 할 계획이다.

state는 좌표 세개, 총 6개의 int  
action은 discrete 5개  
reward는 기본으로 써도 상관없을듯


## 데이터 생성

논문에서 제공하는 npz파일은 s, a, s', done 네가지를 기록한다. default인 하프치타는 4만 step이 주어졌다.


## GridWorld는 안되겠음

그리드월드는 여러 단점이 있음. 일단 연속 액션이라고 하기 애매한 task이기도 하고 이게 imitation으로 해결하기 좋은 문제인지 잘 모르겠음

FoodCollector쪽으로 전환


## 환경 기획

FoodCollector에서 몇가지 수정
1. 카메라 시점변경
2. 액션을 multidiscrete 3, 3으로 (원래 continuous가 들어감)
3. 초록 공 수좀 늘림. 근데 사실 먹는게 의미있는건 아님

## 에이전트 기획

state는 몇갠지 기억은 안나는데 laser sensor 이용
action은 (3, 3)의 출력, 각각 (noop, 앞, 뒤), (noop, 좌, 우)로 구성되어있음
Flatten시켜서 뒤로가는 액션 빼고 6개의 액션으로 구성
reward는 변경 없음

## 데이터 생성

한 에피소드는 5000step, 이 환경은 5step마다 1개의 action이 수행되므로 1000개씩 쌓임. 7000개 모아서 imitation 수행. 어느정도 수행하는걸 볼수있음


## 본격적인 valueDice 구현

논문과 구글 리서치의 코드를 참고해서 구현