from abc import ABC, abstractmethod

class Clipper(ABC):
    """Interface for tensor clipping."""
    def __init__(self):
        pass        

    @abstractmethod
    def _clip(self, tensor):
        """Clips the given tensor."""