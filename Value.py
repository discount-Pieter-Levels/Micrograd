import math
class Value:
    def __init__(self,data,_children=(),_op=''):
        self.data=data
        self.grad=0.0
        self._backward=lambda:None
        self._prev=set(_children)
        self._op=_op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self,other):
        # 1. Ensure 'other' is a Value object
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self,other), '+')
        # 2. Proceed with calculation and graph creation
        
        def _backward():
            # Correct gradients for addition
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = _backward
        return out

    def __neg__(self):
        """Negation: return -self."""
        return self * -1
    
    def __sub__(self, other):

        return self + (-other)

    # It's also good practice to define the reverse operation for robustness:
    def __rsub__(self, other):
        """Reverse Subtraction: return other - self."""
        # This handles cases like '5.0 - Value(2.0)'
        # It's (other) + (-self)
        return self.__neg__() + other 


    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data, (self,other), '*')

        def _backward():
            self.grad+=out.grad*other.data
            other.grad+=out.grad*self.data
        out._backward=_backward
        return out

    def __rmul__(self, other):
        return self*other

    def tanh(self):
 
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1-t**2)*out.grad
        out._backward=_backward

        return out
    def __radd__(self, other):

        return self + other
        
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        # Create a new Value object with the calculated power
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # Apply the power rule: df/dx = n * x^(n-1)
            self.grad += (other * (self.data**(other - 1))) * out.grad

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

# Example to link and compute gradients
x1=Value(2.0)
x2=Value(8.0)
w1=Value(7.0)
w2=Value(-3.0)
b=Value(1.0)

# Build the computation graph
x1w1=x1*w1
x2w2=x2*w2
x1w1x2w2=x1w1*x2w2
n=x1w1x2w2+b

# To get a non-zero gradient, we must call `backward` on the final output.

n.backward()

# Now, the gradients for x1, x2, w1, and w2 are linked and populated.
print(f"n: {n}")
print(f"x1w1: {x1w1}")
print(f"x2w2: {x2w2}")
print("--- Gradients ---")
print(f"x1: {x1}")
print(f"x2: {x2}")
print(f"w1: {w1}")
print(f"w2: {w2}")
print(f"b: {b}")