import numpy as np
import unittest
from first import as_array

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은 지원하지 않습니다.')

        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # y.grad = np.array(1.0) 생략 위해

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
            # x, y = f.input, f.output
            # x.grad = f.backward(y.grad)

                if x.creator is not None:
                    funcs.append(x.creator)
    
    def cleargrad(self):
        self.grad = None

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0] 

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        y = pow(x, 2)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)


if __name__ == '__main__':
    # xs = [Variable(np.array(2)), Variable(np.array(3))]
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    # f = Add()
    # ys = f(xs)
    # y = ys[0]
    # y = f(x0, x1)
    y = add(x0, x1)
    print(y.data)


    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = add(square(x), square(y))
    z.backward()
    print(z.data)
    print(x.grad)
    print(y.grad)

    x = Variable(np.array(3.0))
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)
    
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)

    # 같은 변수 재사용 시 미분 변수 초기화
    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)