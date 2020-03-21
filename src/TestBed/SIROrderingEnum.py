from enum import Enum, IntEnum


class OrderedEnum(Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class SIRS(OrderedEnum):  # SIRS(IntEnum)
    S = 1
    I = 2
    R = 3


if __name__ == '__main__':
    print(SIRS.S > SIRS.R)
    print(SIRS.S < SIRS.R)
    print(SIRS.S == SIRS.S)
    print(SIRS.S == 1)
