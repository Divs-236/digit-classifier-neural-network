import sys;sys.setrecursionlimit(10000)
import numpy as np

"""
Micrograd-inspired Neural network that forms the core of the project.
"""

def _unbrodcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)

    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g

class Value:

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children) 
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), "+")
        def _backward():
            self.grad += _unbrodcast(out.grad, self.data.shape)
            other.grad += _unbrodcast(out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data * other.data, (self,other), "*")
        def _backward():
            self.grad += _unbrodcast(other.data * out.grad, self.data.shape)
            other.grad += _unbrodcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        out = Value(self.data @ other.data, (self, other), "@")
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out


    def sum(self, axis=None, keepdims=False):
        out = Value(self.data.sum(axis = axis, keepdims=keepdims), (self,), "sum")
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += g
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self,other):
        assert isinstance(other, (int,float))  
        out = Value(self.data**other, (self, ), f"**{other}")  
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad

        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + Value(other)


    def dropout(self, p = 0, training=True):
        if not training or p == 0:
            return self

    def tanh(self):
        x = self.data
        t = np.tanh(x)  #(math.exp(2*n) - 1) / (math.exp(2*n) + 1)

        out = Value(t, (self, ), 'tanh' ) 

        def _backward():
            self.grad += (1-t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def log(self):
        x = np.maximum(self.data, 1e-12)
        out = Value(np.log(x), (self,), "log")
        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
    
class Layer:
    def __init__(self, nin, nout, nonlin=True, p=0.0):
        s = (6 / nin) ** 0.5
        self.W = Value(np.random.uniform(-s, s, (nin, nout)))
        self.b = Value(np.zeros((1, nout)))
        self.nonlin = nonlin
        self.p = p
        self.training = True

    def __call__(self, x):
        act = x @ self.W + self.b
        if self.nonlin:
            act = act.relu().dropout(self.p, self.training)
        return act
    
    def parameters(self):
        return [self.W, self.b]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1), p=0.0) for i in range(len(nouts))]

    def __call__(self, x):
        if not isinstance(x, Value):
            x = Value(x)

        for layer in self.layers:
            x = layer(x)

        return x

    def train_mode(self, flag):
        for layer in self.layers:
            layer.training = flag
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
