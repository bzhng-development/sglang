GLOBAL_VAR = "global_value"


class SampleClass:
    def greet(self, name: str) -> str:
        greeting = f"hello {name}"
        return greeting

    def compute(self, x: int) -> int:
        result = x * 2 + 1
        return result

    def uses_global(self) -> str:
        return f"value={GLOBAL_VAR}"


def standalone_function(a: int, b: int) -> int:
    return a + b
