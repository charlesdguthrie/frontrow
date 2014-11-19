#util.py

import os
import time
from functools import wraps


#   Contains utility functions for all of us to use.


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        mymsg = ""
        for key in kwargs.keys():
            if key == "msg":
                mymsg = kwargs[key] + ":"
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print func.__name__ + ":",mymsg,round(end-start,2),"seconds"
        return result
    return wrapper


def ExampleOf_timethis():
    @timethis
    def printhello(*args,**kwargs):
        print "Hello World"
    
    printhello(msg="I'M A KWARG!!!!!")

def RunItTimeIt(func, args=[], returnarg=False, msg=""):
    # This just runs the function func with args and prints a message
    # about how long it took.  You can attach a little message that
    # goes with it.  For example, "Function X completed in [5 seconds]".
    # If no return output required, set returnarg to false, otherwise
    # it will return the output of func.
    #
    # NOTE: args must be a list
    start = time.time()
    if returnarg == True:
        if len(args)==0:
            outargs = func()
        else:
            outargs = func(*args)
    else:
        if len(args)==0:
            func()
        else:
            func(*args)
    end = time.time()
    print msg, round(end-start,2),"seconds"
    if returnarg == True:
        return outargs
        



