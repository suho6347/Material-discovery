import six
from abc import ABCMeta, abstractmethod
import pdb


    
    

# [Chain step 2] Normalizer
class BaseProcessor(six.with_metaclass(ABCMeta)):
    """Abstract processor class from which all processors inherit. Subclasses must implement a ``__call__()`` method."""

    @abstractmethod
    def __call__(self, text):
        """Process the text.

        :param string text: The input text.
        :returns: The processed text or None.
        :rtype: string or None
        """
        return text

class BaseNormalizer(six.with_metaclass(ABCMeta, BaseProcessor)):
    """Abstract normalizer class from which all normalizers inherit.

    Subclasses must implement a ``normalize()`` method.
    """

    @abstractmethod
    def normalize(self, text):
        """Normalize the text.

        :param string text: The text to normalize.
        :returns: Normalized text.
        :rtype: string
        """
        return text

    def __call__(self, text):
        """Calling a normalizer instance like a function just calls the normalize method."""
        return self.normalize(text)


# final step Chain
class Chain(object):
    """Apply a series of processors in turn. Stops if a processors returns None."""

    def __init__(self, *callables):
        self.callables = callables

    def __call__(self, value):
        change_flg_list = []
        values = {"base":value}
        for func in self.callables:
            if value is None: break
            prev_value = value
            value = func(prev_value)
            values[type(func).__name__] = value
            if value != prev_value: change_flg_list.append(1)
            else: change_flg_list.append(0)

        return value, values, change_flg_list