# adapted from https://github.com/whathelll/DeepRLBootCampLabs/tree/master/pytorch/utils
class Epsilon(object):
    def __init__(self, start=1.0, end=0.01, update_increment=0.01):
        self._start = start
        self._end = end
        self._update_increment = update_increment
        self._value = self._start
        self.isTraining = True
    
    def increment(self, count=1):
        self._value = max(self._end, self._value - self._update_increment*count)
        return self
        
    def value(self):
        if not self.isTraining:
            return 0.0
        else:
            return self._value