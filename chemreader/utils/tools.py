def property_getter(func):
    """ Wrapper of the method that tries to return a hidden property of a
    instance. If the property if not defined, call the property constructor
    to generate the property and return it. The property name, hidden property
    name and constructor name must follow the rule below:
    property: foo
    hidden property: _foo
    constructor: _get_foo

    Example:
    class Foo:

        @property
        @property_getter
        def bar(self):
            return self._bar

        def _get_bar(self):
            code to construct the property
            return property_value
    """

    def wrapper(instance):
        try:
            return func(instance)
        except AttributeError:
            getter = instance.__getattribute__("_get_" + func.__name__)
            instance.__dict__["_" + func.__name__] = getter()
            return func(instance)

    return wrapper
