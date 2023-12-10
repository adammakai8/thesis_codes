import time
import sys


def log_runtime_in_debug(func):
    """ Annotáció, ami AOP-t valósít meg, hogy debug módban lehessen logolni az annotált függvény futásidejét """

    def wrapper(*args, **kwargs):
        if sys.gettrace() is not None:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            print(f'[DEBUG] {func.__name__} futásideje: {elapsed_time * 1000}')
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper
