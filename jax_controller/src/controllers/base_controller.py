import numpy as np
import jax
import jax.numpy as jnp

def jaxf1(x, y):
    q = x**3 +3
    z = q**4 + x*y - y
    return z

def jax2(x, y):
    z = 1
    for i in range(int(y)):
        z *= (x+float(i))
    return z

def jax3(x, y):
    return x**y


def f(x, y):
    return 2*x + 3*y + x*y

df = jax.grad(f, argnums=[0,1])
print(df(2.0, 2.0)[1])

def f(w):
    w[0] + w[1]**2 + w[2]**3

w = np.array([3.0, 2.0, 3.0])

print(jax.grad(f)(w))


class BaseController:

    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate