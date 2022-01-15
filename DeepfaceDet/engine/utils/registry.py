import inspect
from functools import partial

from .misc import is_seq_of

def build_from_cfg(cfg, registry, default_args=None):
    """Builds a registry from a configuration dict.

    Args:
        cfg (dict): The configuration.
        registry (dict): The registry.
        default_args (dict): The default arguments.

    Returns:
        object: The built object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    if 'name' not in cfg:
        if default_args is None or 'name' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain a `name` key.'
                f'but got {cfg}\n{default_args}'
            )

    if not isinstance(registry, Registry):
        raise TypeError(f'registry must be a Registry object, but got {type(registry)}')

    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(f'default_args must be a dict or None, but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_name = args.pop('name')
    if isinstance(obj_name, str):
        obj_cls = registry.get(obj_name)
        if obj_cls is None:
            raise KeyError(f'{obj_name} is not registered in {registry.name} registry')
    elif inspect.isclass(obj_name):
        obj_cls = obj_name
    else:
        raise TypeError(f'`name` must be a str or a class, but got {type(obj_name)}')

    try:
        return obj_cls(**args)
    except Exception as e:
        raise type(e)(f'{obj_cls.__name__}:{e}')


class Registry(object):
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={}'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.

        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls=None, force=False):
        """Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name and value is the class itself.
        It can be used as a decorator or a normal function.
        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module
            >>> class ResNet(object):
            >>>     pass
        Example:
            >>> backbones = Registry('backbone')
            >>> class ResNet(object):
            >>>     pass
            >>> backbones.register_module(ResNet)
        Args:
            module (:obj:`nn.Module`): Module to be registered.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
        """
        if cls is None:
            return partial(self.register_module, force=force)

        self._register_module(cls, force=force)
        return cls
