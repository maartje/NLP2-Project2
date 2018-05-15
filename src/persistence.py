import json
import os.path


def save(data, fname):
    with open(fname, "w") as file:
        file.write(str(data))

def load(fname):
    with open(fname, "r") as file:
        data = eval(file.readline())
    return data


