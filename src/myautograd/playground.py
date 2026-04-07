from draw import draw_dot
from extra_tools import print_to_kitty
from value import MLP, Value


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


def mlp_test() -> None:
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    o = n(x)
    print(o)
    dot = draw_dot(o)
    dot.render('computational_graph', format='svg', view=True)


def training_loop(n: MLP, xs, ys, epochs: int, lr: float):
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        ypred = [n(x) for x in xs]
        print("Predictions: ", ypred)
        loss: Value = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
        # Zero grad
        for p in n.parameters():
            p.grad = 0.0

        loss.backward()

        # Updating weights (gradient descent)
        for p in n.parameters():
            p.data += -lr * p.grad

        print("Loss: ", loss.data)

    return loss


def tiny_dataset_test() -> None:
    # dataset
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        # [0.5, 1.0, 1.0],
        # [1.0, 1.0, -1.0],
    ]
    # targets
    ys = [1.0, -1.0, -1.0, 1.0]

    # mlp
    n = MLP(3, [2, 1])
    print("Number of MLP parameters: ", len(n.parameters()))
    
    final_loss = training_loop(n, xs, ys, 10, 0.05)

    dot = draw_dot(final_loss)
    dot.render('loss_graph', format='svg', view=True)

if __name__ == "__main__":
    tiny_dataset_test()


