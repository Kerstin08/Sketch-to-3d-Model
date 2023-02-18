# arguments to use in for argparse
import argparse


def bool(string):
    try:
        if string == 'True':
            return True
        elif string == 'False':
            return False
        else:
            raise argparse.ArgumentTypeError("Given value was neither \"True\" nor \"False\"")
    except:
        raise argparse.ArgumentTypeError("Given value was neither \"True\" nor \"False\"")


def views(s):
    try:
        split = s.split(',')
        list = []
        for index in range(0, len(split), 2):
            list.append((int(split[index]), int(split[index + 1])))
        print(list)
        return list
    except:
        raise argparse.ArgumentTypeError("Views must be tuples of azimuth and elevation angle")
