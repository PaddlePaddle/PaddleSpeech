import inspect


def check_kwargs(func, kwargs, name=None):
    """check kwargs are valid for func

    If kwargs are invalid, raise TypeError as same as python default
    :param function func: function to be validated
    :param dict kwargs: keyword arguments for func
    :param str name: name used in TypeError (default is func name)
    """
    try:
        params = inspect.signature(func).parameters
    except ValueError:
        return
    if name is None:
        name = func.__name__
    for k in kwargs.keys():
        if k not in params:
            raise TypeError(f"{name}() got an unexpected keyword argument '{k}'")
