import typing


class Generator:
    def __init__(self, distribution: typing.Callable):
        self.distribution = distribution
        self.params = distribution()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.distribution.__name__})"

    def __getattr__(self, name: str) -> None:
        if name == "generate":
            raise AttributeError(
                f"Callers should not access {name} directly, use the __call__ method instead"
            )

        return object.__getattribute__(self, name)

    def __call__(self, *args, **kwargs):
        generate = object.__getattribute__(
            self, "generate"
        )  # bypass the __getattr_ restriction, only for this call
        return generate(*args, **kwargs)

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            f"Subclasses of {self.__class__.__name__} must implement the generate method"
        )
