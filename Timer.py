import time

D = {}

def timeit(f):
    def func(*args, **kwargs):
        ts = time.time()
        r = f(*args, **kwargs)
        te = time.time()
        t = te-ts
        if f.__name__ in D:
            D[f.__name__] += t
        else:
            D[f.__name__] = t
        return r
    return func

def Reset():
    for key in D.keys():
        D[key] = 0
