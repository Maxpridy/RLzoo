# 모던 cpp의 문법에 대한 궁금증과 예제












## 실제 std 함수 등등 한줄요약


- future  
std::future_error : 예외. 정의된적 없어도 future로 인한 error를 잡음. std::exception을 상속받음

- variant  
std::bad_variant_access : 예외. variant에 잘못된 요청을 하면 발생함. std::exception을 상속받음

