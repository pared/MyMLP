from typing import List

def pretty_str_list(vec: List[float], precision: int):
    result = "[ "
    for elem in vec:
        result += pretty_str_float(elem, precision) + " "
    return result + "]"

def pretty_str_float(value: float, precision: int):
    return ("%." + ("%df" % precision))%value





