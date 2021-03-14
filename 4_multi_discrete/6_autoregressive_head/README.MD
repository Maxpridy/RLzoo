# autoregressive에 대해서

보통은 다 RNN이 관련되어 있지만 일단은 없이 포함해봄

기본적으로는 action embedding\*state + RNN\*state + (이후의 features) 요런식으로 액션을 출력할때마다 쌓아나간다고 보면될듯

그 정보를 다음으로 전달함


조건부와 유사하지만 최근에 이런 방식으로 많이 사용되는듯함
