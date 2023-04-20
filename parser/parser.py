import inspect
import torch
import re
import json

from collections import defaultdict

classes = set([])
ignored_modules = ['onnx', 'jit', 'ao', 'amp', 'autograd',
                   'utils', 'return_types', 'special', 'backends',
                   'nn.intrinsic', 'nn.Module', 'cpu', 'cuda',
                   'distributed', 'distributions', 'fft',
                   'functional', 'futures', 'fx', 'hub', 'library',
                   'masked', 'multiprocessing', 'nested', 'overrides',
                   'package', 'profiler', 'quantization', 'quasirandom',
                   'serialization', 'sparse', 'storage', 'testing',
                   'torch_version', 'types', 'version']
added_builtins_classes = 'Tensor'
ignored_functions = []

def is_correct_module(module, name, prev_module, prevs):
    return not prev_module \
           or not name.startswith('_') \
           and not name.endswith('_') \
           and not getattr(module, '__file__', '').startswith('_') \
           and 'torch' in getattr(module, '__file__', '') \
           and not getattr(module, '__file__', '') in prevs \
           and prev_module.__name__ in module.__name__ \
           and module.__name__[module.__name__.find('.') + 1:] not in ignored_modules


def is_correct_func(obj, prev):
    return (
        obj.__name__ == '__init__' \
        or (not obj.__name__.startswith('_') \
            and not obj.__name__.endswith('_'))   
        ) \
        and (not prev
            or obj.__module__ and obj.__module__ == 'torch'
            or not '.' in obj.__qualname__
            or obj.__module__ and ('_linalg' in obj.__module__ or '_nn' in obj.__module__)
            or hasattr(prev, '__qualname__') and hasattr(obj, '__qualname__') 
            and '.'.join(obj.__qualname__.split('.')[:-1]) == prev.__qualname__)

# \
#     and (not prev
#         or obj.__module__ and obj.__module__ == 'torch'
#         or not '.' in obj.__qualname__
#         or obj.__module__ and ('_linalg' in obj.__module__ or '_nn' in obj.__module__)
#         or hasattr(prev, '__qualname__') and hasattr(obj, '__qualname__') 
#         and '.'.join(obj.__qualname__.split('.')[:-1]) == prev.__qualname__)
           


def is_correct_class(obj, prev):
    return not obj.__name__.startswith('_') \
           and not obj.__name__.endswith('_') \
           and not obj.__module__ + '.' + obj.__name__ in classes \
           and (not prev or obj.__module__ == prev.__name__)
           

def get_builtins_args(obj):
   r = re.compile('\(.+\)')
   if obj.__doc__:
        func = re.search(r, obj.__doc__)
        if func:
            named_args = {}
            args = [i.strip() for i in func[0][1:-1].split(',')]
            for arg in args:
                named_arg = {'Type': None, 'Default': None}
                t = re.split(':', arg)
                d = re.split('=', t[-1])
                if len(t) > 1:
                    named_arg['Type'] = t[-1].strip()
                else:
                    t[0] = d[0]
                if len(d) > 1:
                    named_arg['Default'] = d[-1].strip()
                named_args[t[0]] = named_arg
            
            return named_args


def get_args(func):
    args, varargs, keywords, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(func)
    
    if args and defaults:
        defaults = dict(zip(args[len(args) - len(defaults):], defaults))
    if varargs: args.extend(['*' + varargs])
    if keywords: args.extend(['**' + keywords])
    if kwonlydefaults: args.extend(kwonlydefaults)
    
    if kwonlydefaults: 
        if defaults:
            defaults.update(kwonlydefaults)
        else:
            defaults = kwonlydefaults
    
    named_args = defaultdict(dict)
    for arg in args:
        t = str(annotations[arg]) if annotations and arg in annotations.keys() else None
        d = str(defaults[arg]) if defaults and arg in defaults.keys() else None
        named_args[arg] = {'Type': t,
                           'Default': d}
                   
    return named_args


def add_param(name, obj, hierarchy, prev, prevs, num):
    if inspect.ismodule(obj):
        if is_correct_module(obj, name, prev, prevs):
            new_prevs = prevs.copy()
            new_prevs.append(obj.__file__)
            hierarchy['Modules'][name] = defaultdict(dict)
            hierarchy['Modules'][name]['Doc'] = str(obj.__doc__) if obj.__doc__ else None
            for n, o in inspect.getmembers(obj):
                add_param(n, o, hierarchy['Modules'][name], obj, new_prevs, num+1)
    elif inspect.isclass(obj):
        if is_correct_class(obj, prev):
            hierarchy['Classes'][name] = defaultdict(dict)
            hierarchy['Classes'][name]['Doc'] = str(obj.__doc__) if obj.__doc__ else None
            classes.add(obj.__module__ + '.' + obj.__name__)
            for n, o in inspect.getmembers(obj):
                add_param(n, o, hierarchy['Classes'][name], obj, prevs, num + 1)
    elif inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
        if is_correct_func(obj, prev):
            arg_dict = {}
            hierarchy['Functions'][name] = defaultdict(dict)
            arg_dict['Doc'] = str(obj.__doc__) if obj.__doc__ else None
            arg_dict['Args'] = {}
            if not inspect.isbuiltin(obj):
                arg_dict['Args'] = get_args(obj)
            else:
                arg_dict['Args'] = get_builtins_args(obj)
            hierarchy['Functions'][name] = arg_dict


def clear(torch_hierarchy):
    cleared_torch_hierarchy = defaultdict(dict)
    for i, j in torch_hierarchy['Modules']['torch'].items():
        cleared_torch_hierarchy[i] = j
        if i == 'Classes':
            cleared_classes = defaultdict(dict)
            for i1, j1 in j.items():
                if 'Tensor' in i1:
                    cleared_classes[i1] = j1
            
            cleared_torch_hierarchy[i] = cleared_classes
    
    return cleared_torch_hierarchy



def save(torch_hierarchy, cleared_torch_hierarchy):
    with open('hierarchy/torch_new.json', 'w') as f:
        json.dump(torch_hierarchy, f, indent='  ')
        
    with open('hierarchy/cleared_torch_new.json', 'w') as f:
        json.dump(cleared_torch_hierarchy, f, indent='  ')


if __name__ == '__main__':
    torch_hierarchy = defaultdict(dict)
    add_param('torch', torch, torch_hierarchy, None, [], 0)
    cleared_torch = clear(torch_hierarchy)
    save(torch_hierarchy, cleared_torch)