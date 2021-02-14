# 모듈과 pybind11 : 2021.02.02

## 이 단락을 생성한 이유

torchbeast를 보다가 넘어왔다. 언제곤 꼭 cpp를 라이브러리로 사용해보자는 생각을 했었지만 swig나 다른 방법들은 번잡해보여서 하지 않았었는데 이번에 torchbeast에 구현되어 있는것이 깔끔하고 조금만 응용하면 간단한 예제를 만들 수 있을것같아 시도하게 되었다. 이번 기회에 조금 시도해보자.

pytorch에도 포함되어 있다. 나아가서 pytorch 구현 자체에 대한 이해도 높일 수 있는 기회라고 생각한다. 

## 목적

이 단락에선 파이썬 모듈 시스템에 대해 알아보고, pybind11로 cpp를 간편하게 python 모듈로 만드는것들을 테스트해보자.

사실 이 부분은 굉장히 안일하게 넘어가는 경우가 많았는데 이번 기회에 실습으로 충분히 알게 될면 좋을것이다.

겸사겸사 pybind11로 간단한 모듈을 만들어서 파이썬에서 불러오는 테스트까지 실행해보자.

참고자료  
https://docs.python.org/3/extending/building.html


## python setup.py install/build/test/...

파이썬의 모듈은 기본적으로 setup.py를 통해서 패키지에 대한 여러가지 관리를 할 수 있도록 되어있다(사실 파이썬의 많은 부분들이 그렇듯이 약속일 뿐이다. 이름이 굳이 setup.py가 아니여도 되긴된다). 그 내부의 구현에는 아래에서 다룰 setuptools를 사용한다.

install이든 build든 특별할것 없는 python의 경우엔 파이썬 파일들이 단순히 묶이게 되고, Extension으로 cpp를 묶는 경우엔 동적 라이브러리인 .so/.pyd가 생성되게 되고 이것을 import해서 사용하는 방식이 된다.

기타 test, develop 등이 있는데 필요할때마다 알아보도록 하자

## setuptools

파이썬에서는 setuptools 패키지로 프로젝트를 관리할 수 있다. 이것을 이용해서 다른사람이 이 모듈을 쉽게 설치할수도 있고, 빌드와 테스트도 할 수 있다.

여기선 굳이 README를 세부적으로 나누지 않고 여기서 계속 이야기하겠다.

일단 첫번째 예제에서는 from setuptools import setup만 사용해도 된다. 
두번째 예제에서는 Extension을 포함해야 하는데 

참고자료  
https://setuptools.readthedocs.io/en/latest/setuptools.html


## 단순 python
(단순_python 폴더 참고)

말 그대로 정말 단순하게 파이썬으로 만들어본것이다. 

저 폴더에서 python setup.py install 혹은 pip install -e .를 치고나서
(파이썬 3.8을 권장)

```
from this_is_package_name_wow.import_this_one import Test1

t = Test1()
t.v # 1
```
를 하게되면 모듈이 불러와진다. 내 인터프리터의 경로에 이 모듈이 설치가 된것이다. 

이 부분에서 작성한건 사실상 setup.py가 다인데, 내용을 보면 특별할것은 없다. 

기억해둘만한 점은 this_is_package_name_wow의 하위 폴더들은 포함이 안된다.

여기까진 쉽다.


## 단순 cpp

https://github.com/pybind/python_example
이 repo는 pybind11에서 제공하는 setuptools 공식 예제인데 아주 간단하다.

단락이 단순 cpp이므로 저걸 보는걸로 넘어가도 되겠다.



## pybind 활용

모던 cpp 복습하느라고 며칠동안 좀 봤다. 역시 장난이 아니다.. 최근 언어들에 대한 패러다임, 마엘스트롬 어쩌구 저쩌구가 다 담긴 최신언어이다. 물론 진짜 최신언어들에 비해선 뭔가 좀 손색이 있는것같지만 어쨋든 계속 모ㅡ던 해지고있다.

torchbeast에는 nest라는 간단한 모듈이 있는데 이것과 polybeast를 좀 살펴봤다.

