import numpy as np
import pickle
import os

#================================================================================================================

# Two basic functions to make storing objects to file quick and easy, using pickle

def save_object_to_file(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file = open(filename,"wb")
    pickle.dump(obj, file)
    file.close()

def load_object_from_file(filename):
    file = open(filename,"rb")
    obj = pickle.load(file)
    file.close()
    return obj

#================================================================================================================