import numpy as np
# f(x,y) = (x-2)**2 +3(y+1)**2 

def grad(u):
    x,y = u
    return np.array(
        [2*(x-2),
         6*(y+1)])

def hessian(u):
    x, y = u
    return np.array([[2, 0],
                     [0, 6]])

def newton_optim(u_0, grad_fn, hessian_fn, tol=1e-6, max_iter=100):
    u = u_0
    for _ in range(max_iter):
        grad = grad_fn(u)
        hessian = hessian_fn(u)
        t = np.linalg.solve(hessian, -grad)
        u = u + t
    return u

if __name__ == "__main__":
    u_0 = np.array([0.0, 0.0])
    u_optim = newton_optim(u_0, grad, hessian)
    print(u_optim)
