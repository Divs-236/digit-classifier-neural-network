import random
import sys;sys.setrecursionlimit(10000)
import math


class Value:

    __slots__ = ("data","grad","_backward","_prev","_op","label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
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
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data * other.data, (self,other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

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
    
    def tanh(self):
        x = self.data
        t = math.tanh(x)#(math.exp(2*n) - 1) / (math.exp(2*n) + 1)

        out = Value(t, (self, ), 'tanh' ) 

        def _backward():
            self.grad += (1-t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def log(self):
        x = max(self.data, 1e-12)
        out = Value(math.log(x), (self,), "log")
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

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

def dot_const(ws, xs, b):

    out = Value(sum([w.data * x for w, x in zip(ws, xs)]) + b.data, tuple(ws) + (b,), "dot")

    def _backward():
        g = out.grad
        for w, x in zip(ws, xs):
            w.grad += x * g

        b.grad += g
    out._backward = _backward
    return out


def dot_value(ws, xs, b):
    out = Value(sum([w.data * x.data for w, x in zip(ws, xs)]) + b.data, tuple(ws) + tuple(xs) + (b,), "dot")

    def _backward():
        g = out.grad
        for w, x in zip(ws, xs):
            w.grad += x.data * g
            x.grad += w.data * g

        b.grad += g
    out._backward = _backward
    return out



class Neuron:
    def __init__(self, nin, nonlin=True):
        s = nin ** -0.5
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin

    def __call__(self, x):

        if isinstance(x[0], Value):
            act = dot_value(self.w, x, self.b)
        else:
            act = dot_const(self.w, x, self.b)

        return act.relu() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
