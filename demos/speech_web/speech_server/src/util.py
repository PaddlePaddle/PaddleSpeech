import random

def randName(n=5):
    return "".join(random.sample('zyxwvutsrqponmlkjihgfedcba',n))

def SuccessRequest(result=None, message="ok"):
    return {
        "code": 0,
        "result":result,
        "message": message
    }

def ErrorRequest(result=None, message="error"):
    return {
        "code": -1,
        "result":result,
        "message": message
    }