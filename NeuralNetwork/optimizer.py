from autograd.misc import flatten
from autograd.wrap_util import wraps
from autograd import grad
import autograd.numpy as np


## References -> Autograd misc/optimizers,py 
## https://github.com/HIPS/autograd/blob/01eacff7a4f12e6f7aebde7c4cb4c1c2633f217d/autograd/misc/optimizers.py#L16


def unflatten_optimizer(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""
    @wraps(optimize)
    def _optimize(grad, x0, callback=None, *args, **kwargs):
        _x0, unflatten = flatten(x0)
        _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
        if callback:
            _callback = lambda x, i, g: callback(unflatten(x), i, unflatten(g))
        else:
            _callback = None
        return unflatten(optimize(_grad, _x0, _callback, *args, **kwargs))

    return _optimize


@unflatten_optimizer
def param_update(grad, params, callback=None, learning_rate=0.0001, num_iter=1000):
    
    for i in range(num_iter):
        # print(i)
        der = grad(params, i)
        # print(learning_rate)
        params = params - learning_rate*der

    return params