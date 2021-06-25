import numpy as np
import unittest

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
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = pow(x, 2)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

if __name__ == '__main__':
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)

    # assert y.creator == C
    assert y.creator.input == b
    # assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    # assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    # b.grad = C.backward(y.grad)
    # a.grad = B.backward(b.grad)
    # x.grad = A.backward(a.grad)
    # print(x.grad)

    # C = y.creator
    # b = C.input
    # b.grad = C.backward(y.grad)

    # B = b.creator
    # a = B.input
    # a.grad = B.backward(b.grad)

    # A = a.creator
    # x = A.input
    # x.grad = A.backward(a.grad)
    # print(x.grad)

    # y.backward()
    # print(x.grad)

    y = square(exp(square(x)))
    y.backward()
    print(x.grad)

    unittest.main()