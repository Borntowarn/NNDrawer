import inspect
import torch

def get_args(func):
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return args, defaults

def add_to_hierarchy(name, obj, hierarchy):
    if inspect.ismodule(obj) or inspect.isclass(obj):
        hierarchy[name] = {}
        for n, o in inspect.getmembers(obj):
            add_to_hierarchy(n, o, hierarchy[name])
    elif inspect.isfunction(obj):
        args, defaults = get_args(obj)
        arg_dict = {}
        if args:
            for i, arg in enumerate(args):
                arg_dict[arg] = defaults[i] if defaults and len(defaults) > i else None
        hierarchy[name] = arg_dict

torch_hierarchy = {}
for name, obj in inspect.getmembers(torch):
    if inspect.ismodule(obj) or inspect.isclass(obj):
        torch_hierarchy[name] = {}
        for n, o in inspect.getmembers(obj):
            add_to_hierarchy(n, o, torch_hierarchy[name])
    elif inspect.isfunction(obj):
        args, defaults = get_args(obj)
        arg_dict = {}
        if args:
            for i, arg in enumerate(args):
                arg_dict[arg] = defaults[i] if defaults and len(defaults) > i else None
        torch_hierarchy[name] = arg_dict

print(torch_hierarchy) 
