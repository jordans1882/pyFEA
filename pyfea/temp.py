"""The template module.

This module has example classes to showcase how to write classes, methods and
documentation for the python-template package. See the Template class and the
template function.
"""


class Temp:
    """
    A class to represent a template.

    Attributes
    ----------
       val : numeric
           The answer to everything
       val2 : numeric
           I'm feeling lucky
    """

    def __init__(self, val=42, val2=7):
        """Construct all the necessary attributes for the template object.

        Parameters
        ----------
            val : numeric
                The answer to everything
            val2 : numeric
                I'm feeling lucky
        """
        self.val = val
        self.val2 = val2
        self.val3 = None

    def hello_world(self):
        """Print hello world."""
        print("Hello world!!")
        print(f"The secret number is: {self.val}")

    def hello_world_v2(self):
        """Print hello world again."""
        print("Hello world!!")
        print(f"The other secret number is: {self.val2}")

    def identity(self, val: float) -> float:
        """Return self function (identity)."""
        self.val3 = val
        return val


def temp():
    """Print hello world."""
    print("Hello World!!")
