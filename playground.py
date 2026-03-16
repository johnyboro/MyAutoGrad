from draw import draw_dot
from extra_tools import print_to_kitty
from value import Value


def simple_test() -> None:
    a = Value(3.0, label='a')
    b = Value(2.0, label='b')
    c = Value(-2.0, label='c')

    d = a * b
    d.label = 'd'
    e = d + c
    e.label = 'e'
    f = Value(10.0, label='f')
    L = e * f
    L.label = 'L'

    print_to_kitty(draw_dot(L))

def backpropagation_test() -> None:
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    b = Value(6.8813735870195432, label='b')

    # simple neuron
    x1w1 = x1*w1
    x1w1.label = 'x1*w1'

    x2w2 = x2*w2
    x2w2.label = 'x2*w2'

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1*w1 + x2*w2'

    n = x1w1x2w2 + b
    n.label = 'n'

    # o = n.tanh()\
    e = (2*n).exp()
    o = (e - 1) / (e + 1)
    o.label = 'o'

    # backpropagation
    o.backward()

    dot = draw_dot(o)
    dot.render('computational_graph', format='svg', view=True)

if __name__ == "__main__":
    backpropagation_test()



