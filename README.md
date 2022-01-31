# RL Zoo

이 repo에서는 강화학습으로 실험해본 내역들을 정리합니다.  



## zoo 목록

0. 강화학습, 딥러닝의 기초가 되는 구현
1. Unity ML-agents 관련
2. NLE로 간단한 여러가지 테스트
3. gfootball 환경(캐글)에서 7등 이기기
4. multi-discrete에 대한 정리
5. multi-process에 대한 정리
6. model_base_or_free
7. on_off_policy
8. distillation


## 현재 탐색중인 위치(stack)

/8. distillation
/5. pybind11의 gil을 살펴보고 작성  
/1. 모던 cpp  -> 따로 레포 만듬  
/1. 기초 - 모듈과 pybind  
/5. 멀티프로세스 - 토치비스트  


## todo

autoregressive head -> 다양한 방식으로 시도해보기  
td lambda without vtrace in value -> 토치비스트로 한번 테스트해볼만  
upgo  
위의 것들과 연관이 있는 람다에 대해서 살펴보기. upgo와 bootstraping의 효과에 대해서 간단하지만 어떤 효과를 가져오는지에 대한 생각 -> https://github.com/deepmind/trfl/blob/08ccb293edb929d6002786f1c0c177ef291f2956/trfl/sequence_ops.py  
얼라인러더와 알파스타의제약. sequential한 지도학습적인 제약에 대해  