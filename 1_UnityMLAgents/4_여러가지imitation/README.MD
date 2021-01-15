# 여러가지 imitation

이미테이션에는 역시 ml-agents만한게 없어서 mlagents의 food collector로 시도했다.

데이터셋을 어떻게 만들었냐면 사실 아주 잘먹는 데이터셋을 생성하진 않았다. 이 환경은 강화학습이 훨씬 잘하기때문에... 나는 외곽을 좀 잘 도는 액션을 위주로 action을 만들었다.

일단 학습된 모델부터 보자

## catboost

캣부스트는 이런 느낌이다. 전반적인 가장 올바른 정답을 고르는 느낌에 가깝다. max로 inference를 했기때문에 안정적인 느낌이다.

## BC(pytorch)

BC는 이런 느낌이다. 갈팡질팡 하지만 여러가지 상황을 고려하려는듯한 모습이다.

## catboost모델을 KL로 학습(pytorch)

KL로 catboost모델을 학습한 모델이다. 위랑 비슷하다. 



