from __future__ import annotations
from graphviz import Digraph

from extra_tools import print_to_kitty

class Value:
    def __init__(self, data, _children=(), _op='', label='') -> None:
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other) -> Value:
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other) -> Value:
        return Value(self.data * other.data, (self, other), '*')


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid+n._op, label=n._op)
            dot.edge(uid+n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def test() -> None:
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

    f.grad = 4.0
    e.grad = 10.0
    print_to_kitty(draw_dot(L))


if __name__ == "__main__":
    test()


