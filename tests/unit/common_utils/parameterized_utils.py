import json
from itertools import product
import os

from parameterized import param, parameterized

#def get_asset_path(*paths):
    #"""Return full path of a test asset"""
    #return os.path.join(_TEST_DIR_PATH, "assets", *paths)

#def load_params(*paths):
    #with open(get_asset_path(*paths), "r") as file:
        #return [param(json.loads(line)) for line in file]

#def load_effects_params(*paths):
    #params = []
    #with open(get_asset_path(*paths), "r") as file:
        #for line in file:
            #data = json.loads(line)
            #for effect in data["effects"]:
                #for i, arg in enumerate(effect):
                    #if arg.startswith("<ASSET_DIR>"):
                        #effect[i] = arg.replace("<ASSET_DIR>", get_asset_path())
            #params.append(param(data))
    #return params

def _name_func(func, _, params):
    strs = []
    for arg in params.args:
        if isinstance(arg, tuple):
            strs.append("_".join(str(a) for a in arg))
        else:
            strs.append(str(arg))
    # sanitize the test name
    name = "_".join(strs)
    return parameterized.to_safe_name(f"{func.__name__}_{name}")


def nested_params(*params_set, name_func=_name_func):
    """Generate the cartesian product of the given list of parameters.

    Args:
        params_set (list of parameters): Parameters. When using ``parameterized.param`` class,
            all the parameters have to be specified with the class, only using kwargs.
    """
    flatten = [p for params in params_set for p in params]

    # Parameters to be nested are given as list of plain objects
    if all(not isinstance(p, param) for p in flatten):
        args = list(product(*params_set))
        return parameterized.expand(args, name_func=_name_func)

    # Parameters to be nested are given as list of `parameterized.param`
    if not all(isinstance(p, param) for p in flatten):
        raise TypeError("When using ``parameterized.param``, " "all the parameters have to be of the ``param`` type.")
    if any(p.args for p in flatten):
        raise ValueError(
            "When using ``parameterized.param``, " "all the parameters have to be provided as keyword argument."
        )
    args = [param()]
    for params in params_set:
        args = [param(**x.kwargs, **y.kwargs) for x in args for y in params]
    return parameterized.expand(args)
