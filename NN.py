import random
class Neuron:
    def __init__(self,nin):
        self.w =[Value(random.uniform(-1,1))for _ in range (nin)]
        self.b=Value(random.uniform(-1,1))
    def __call__(self,x):
        #x_values = [Value(xi) for xi in x]
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = act.tanh()
        return out
    def params(self):
        return self.w+[self.b]

class layer:
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        outs =  [n(x) for n in self.neurons]
        return outs[0] if len(outs) ==1 else outs
    def params(self):
        param=[]
        for neuron in self.neurons:
            ps=neuron.params()
            param.extend(ps)
        return param
class MLP:
    def __init__(self,nin,nouts):
        sz = [nin]+nouts
        self.layers=[]
        for i in range(len(nouts)):
            self.layers.append(layer(sz[i],sz[i+1]))

    def __call__(self, x):
        for layer in self.layers:
            x=layer(x)
        return x
    def param(self):
        p=[]
        for layer in self.layers:
            ps = layer.params()
            p.extend(ps)
        return p
        
x =[2.0,3.0,-1.0]
n=MLP(3,[4,4,1])
n(x)
xs =[
    [-7.0,-9.0,-5.0],
    [10.0,-7.0,-5.0],
    [8.0,5.0,3.0],
    [-0.9,7.0,10.0]
]

ys =[1.0,-1.0,-1.0,1.0]
ypred = [n(x) for x in xs]

for k in range(10):
    ypred = [n(x) for x in xs]
    loss=sum((yout-yget)**2 for yout,yget in zip(ys,ypred))

    for p in n.param():
        p.grad=0.0
    loss.backward()

    for p in n.param():
        p.data+=-0.05*p.grad

    print(k,loss.data)