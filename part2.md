# 출처
> 밑바닥부터 시작하는 딥러닝 3 : https://www.hanbit.co.kr/store/books/look.php?p_code=B6627606922

> 소스코드 : https://github.com/WegraLee/deep-learning-from-scratch-3


# [DeZero] 2. 기능 확장 및 정리

- 목차
    - [1. 계산 그래프](#1-계산-그래프)
    - [2. 메모리 관리와 절약](#2-메모리-관리와-절약)
    - [3. 유용한 기능 추가](#3-유용한-기능-추가)
    - [4. 개선된 구현](#4-개선된-구현)
    - [5. 패키징 및 테스트](#5-패키징-및-테스트)
    - [6. 부록](#6-부록)


## 1. 계산 그래프
### 가변 길이 인수
1. 인수와 반환값의 타입을 리스트로 바꾸자
   * 리스트 내포(list comprehension)
     ```python
     xs = [x.data for x in inputs]
     ```
2. 가변 길이 인수
   * Asterisk(\*) 기호를 사용하여 함수의 parameter 를 표시함
     ```python
     def __call__(self, *input): # 가변인자 input
     ```
   * 리스트 언팩(list unpack)
     ```python
     self.forward(*xs) # self.forward(x0, x1) 와 같은 동작
     ```

### 같은 변수 반복 사용
* 인플레이스 연산(in-place operation) : 복사하지 않고 메모리의 값을 직접 덮어 쓰는 연산
  ```python
  x += x    # 덮어쓰기(overwrite) : x 의 객체 ID 가 변하지 않음
  x = x + x # 복사(copy) : x 의 객체 ID 가 달라짐
  ```
* 미분값 재설정
  * 서로 다른 두가지 미분 계산을 수행하려고 하는데, Variable 인스턴스 x를 재사용하고 싶은 경우 x 의 미분값을 초기화해준다.

### 복잡한 계산 그래프
* 위상(topology) : 그래프의 '연결된 형태'
* 위상 정렬(topology sort)
  * 함수와 변수의 세대(generation)를 설정하여 정렬하자.
  * 최근 세대의 함수부터 꺼내도록 하여 올바른 순서로 역전파가 이루어지도록 하자.


## 2. 메모리 관리와 절약
### 메모리 관리와 순환 참조
* 메모리 누수(memory leak)
* 메모리 부족(out of memory)
* 파이썬(CPython)의 메모리 관리 방식
  1. 참조 카운트 : 참조(reference)를 세고, 참조 수가 0이 되는 즉시 해당 객체를 메모리에서 삭제하는 방식
     * 한계 : 순환 참조(circular reference)일 때, 대입 해제(a = None)를 해도 참조 카운트가 0이 되지 않는 문제를 해결할 수 없다.
  2. 세대별 가비지 컬렉션(gernerational garbage collection) : 세대(generation)을 기준으로 쓸모없어진 객체(garbage)를 회수(collection)하는 방식
     * 메모리가 부족해지는 시점에 파이썬 인터프리터에 의해 자동으로 호출되거나 명시적으로 호출
       ```python
       gc.collect()
       ```
* DeZero에 존재하는 순환 참조
  * (input)Variable -> Function -> (output)Variable
  * Function 인스턴스는 두 개의 Variable 인스턴스를 참조한다.
  * 이때, output Variable 인스턴스가 creator로 Function 인스턴스를 참조하기 때문에 순환 참조 관계가 형성된다.
  * 따라서 Function이 output Variable을 약한 참조로 가리키도록 변경한다.
* weakref 모듈
  * 약한 참조(weak reference) : 다른 객체를 참조하되 참조 카운트는 증가시키지 않는 기능
    * 참조된 데이터에 접근하기 위해서는 `()`를 붙여야 한다.
      ```python
      a = np.array([1,2,3])
      b = weakref.ref(a)
      print(b()) # [1,2,3]
      ```
  * but, IPython과 주피터 노트북(Jupyter Notebook) 등의 인터프리터를 인터프리터 자체가 사용자가 모르는 참조를 추가로 유지한다.

### 메모리 절약 모드
* 필요없는 미분값 삭제 : 미분값이 필요한 변수만 저장하고, 중간 변수들의 미분값은 메모리에서 바로바로 제거될 수 있도록 한다.
* 학습(training) 시에는 미분값을 구해야 하지만(순전파+역전파), **추론(inference) 시에는 단순히 순전파만 진행**한다. 따라서 중간 계산 결과를 곧바로 버리면 메모리 사용량을 크게 줄일 수 있다.
  * Config 클래스 활용 : 역전파 활성/비활성 모드 설정
    * 설정 데이터는 단 한군데에만 존재하는 것이 좋다. 따라서 **인스턴스화하지 않고 클래스 상태로 이용**한다.
  * with 문을 이용한 모드 설정
    * with 블록 안에서만 '역전파 비활성 모드'
    * with 블록을 벗어나면 일반 모드, 즉 '역전파 활성 모드'


## 3. 유용한 기능 추가
### 변수 사용성 개선
* 변수 이름을 붙이면, 계산 그래프 등을 시각화할 때 표시할 수 있다.
* `@property` 를 붙이면, 메서드를 인스턴스 변수처럼 사용할 수 있게 된다.
* 특별한 의미를 지닌 메서드
  * `__init__` : 초기화
  * `__len__` : 길이 반환
  * `__repr__` : print함수가 출력해주는 문자열

### 연산자 오버로드
* 연산자 오버로드(operator overload)
  * `+`, `*`와 같은 연산자를 사용하기 위해서 `__add__`, `__mul__` 과 같은 특수 메서드를 정의한다.
    * `__mul__(self, other)`  : (Variable) * (other) 인 경우 호출됨
    * `__rmul__(self, other)` : (other) * (Variable) 인 경우 호출됨
* 변수 타입에 따라서도 계산 가능하게 만들기
  * `Variable` and `Variable`
  * `Variable` and `ndarray`
  * `Variable` and `int`
  * `Variable` and `float`


## 4. 개선된 구현
```python
import numpy as np
import weakref
import contextlib

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data # 통상값
        self.name = name # 변수 이름
        self.grad = None # 미분값
        self.creator = None # 함수와의 관계 (핵심☆)
        self.generation = 0 # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 세대 기록
    
    def cleargrad(self):
      self.grad = None # 미분값 재설정
    
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # dy/dy = 1

        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation) # 넣을 때마다 정렬

        #funcs = [self.creator]
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx # 전파되는 미분값의 합 구하기
                if x.creater is not None:
                    #funcs.append(x.creator)
                    add_func(x.creator)
            
            if not retain_grad: # 중간 변수의 미분값을 저장하지 않는다면
                for y in f.outputs:
                    y().grad = None # 중간 변수의 미분값을 모두 None으로 재설정
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)

# ===========================================================
class Config:
    enable_backprop = True # 역전파 활성/비활성 모드

@contextlib.contextmanager
def using_config(name, value):
    # getattr, setattr : 파이썬 내장함수
    old_value = getattr(Config, name) 
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

# ===========================================================
def as_array(x):
    '''0차원 ndarray 인스턴스는 계산 결과의 데이터 타입이 달라지기 때문에 조정해야 한다.'''
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            for output in outputs:
                output.set_creator(self) # 함수와의 관계 설정
            self.inputs = inputs # 입력 변수 기억
            self.outputs = [weakref.ref(output) for output in outputs] # 출력 변수(를 약한 참조로) 기억
            self.generation = max([x.generation for x in inputs]) # 입력 변수 중 가장 최근 세대로 함수의 세대 설정
        
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

# === 연산자 오버로드 =======================================
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def setup_variable():
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
```


## 5. 패키징 및 테스트
### 용어
* 모듈(Module) : 파이썬 파일, 다른 파이썬 프로그램에서 `import`하여 사용
* 패키지(Package) : 여러 모듈을 묶은 것, 디렉터리 안에 모듈 추가
* 라이브러리(Library) : 여러 패키지를 묶은 것, 하나 이상의 디렉터리

### 패키징
* `dezero` (패키지)
  * [`__init__.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/__init__.py)
    * 모듈을 임포트할 때 가장 먼저 실행되는 파일
    * dezero 패키지를 임포트하면, `__init__.py` 파일에서 `from dezero.core_simple import Variable` 가 실행되므로, Variable 클래스를 바로 임포트할 수 있다. (`from dezero import Variable`)
  * [`core_simple.py`](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/dezero/core_simple.py) (모듈)
    * DeZero 의 핵심, 나중에 `core.py`가 최종

* [패키지 임포트 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step23.py)
  * `if '__file__' in globals():`
    * `__file__`이라는 전역 변수가 정의되어 있는지 확인한다.
    * 터미널에서 python 명령으로 실행하면 `__file__`변수가 정의되어 있다.
    * 파이썬 인터프리터의 인터랙티브 모드와 구글 코랩 등의 환경에서 실행하는 경우에는 정의되어 있지 않다.
  * `sys.path.append()` : `sys.path`에는 파이썬 라이브러리들이 설치되어있는 디렉터리들이 들어있다. 따라서 모듈 검색 경로를 추가하는 코드이다.
    * `pip install dezero` 명령으로 DeZero가 단순 디렉터리가 아닌 패키지로 설치된 경우에는 DeZero 패키지가 파이썬 검색 경로에 추가된다.

### 테스트
* 최적화 문제의 테스트 함수 : 다양한 최적화 기법이 '얼마나 좋은가'를 평가하는 데 사용되는 함수, '벤치마크'용 함수
* [위키백과의 'Test functions for optimization'페이지 참고](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
* [복잡한 함수의 미분 테스트 예제](https://github.com/WegraLee/deep-learning-from-scratch-3/blob/master/steps/step24.py)
  * Sphere : $z = x^2 + y^2$
  * matyas (called 마차시) : $z = 0.26(x^2 + y^2) - 0.48xy$
  * Goldstein-Price : $z = [1+(x+y+1)^2(19-14x+3x^2-14y+6xy+3y^2)][30+(2x-3y)^2(18-32x+12x^2+48y-36xy+27y^2)]$


## 6. 부록
### Define-and-Run(정적 계산 그래프 방식)
* 초기 대부분의 방식, 텐서플로, 카페, CNTK 등이 도입
* 특징
  * 계산 그래프를 사용자가 정의하고, 프레임워크는 주어진 그래프를 컴퓨터가 처리할 수 있는 형태로 변환(컴파일)하여 데이터를 흘려보낸다.
  * 기호 프로그래밍(symbolic programming) : 실제 수치가 아닌 기호를 대상으로 프로그래밍, 실제 데이터를 흘려보내기 전까지 실제 계산이 이루어지지 않는다.
  * 도메인 특화 언어(Domain-Sepcific Language, DSL) : 프레임워크 자체의 규칙들로 이루어진 언어, 한마디로 파이썬 위에서 동작하는 새로운 프로그래밍 언어
    * 미분 가능 프로그래밍(differentiable programming) : 미분을 하기 위해 설계된 언어 (딥러닝 프레임워크 등)
* 장점
  * 계산 그래프를 최적화하면 성능도 최적화된다.
  * 파이썬이 아닌 다른 환경에서 데이터를 흘려보낼 수 있다.
* 예시
  ```python
  import tensorflow as tf
  
  flg = tf.placeholder(dtype=tf.bool)
  x0 = tf.placeholder(dtype=tf.float32)
  x1 = tf.placeholder(dtype=tf.float32)
  y = tf.cond(flg, lambda: x0+x1, lambda: x0*x1) # if문 역할, flg값에 따라 처리
  ```

### Define-by-Run(동적 계산 그래프 방식)
* 2015년 체이너(Chainer)에 의해 처음 제창, 파이토치, MXNet, DyNet, 텐서플로(2.0 이상 기본값) 등이 도입
* 특징
  * Define-and-Run과 달리 '데이터 흘려보내기'와 '계산 그래프 구축'이 동시에 이루어진다.
  * 사용자가 데이터를 흘려보낼 때(수치 계산을 수행할 때) 자동으로 계산 그래프를 구성하는 '연결(참조)'를 만든다.
* 장점
  * 도메인 특화 언어를 배우지 않아도 된다. (파이썬 디버거 사용 가능)
