# arguments to use in for argparse
import argparse
import typing
from source.util import data_type


def p_bool(
        input_bool: str
) -> bool:
    try:
        if input_bool == 'True':
            return True
        elif input_bool == 'False':
            return False
        else:
            raise argparse.ArgumentTypeError("Given value was neither \"True\" nor \"False\"")
    except:
        raise argparse.ArgumentTypeError("Given value was neither \"True\" nor \"False\"")


def p_views(
        input_views: str
) -> list[typing.Tuple[int, int]]:
    try:
        split = input_views.split(',')
        view_list = []
        for index in range(0, len(split), 2):
            view_list.append((int(split[index]), int(split[index + 1])))
        return view_list
    except:
        raise argparse.ArgumentTypeError("Views must be tuples of azimuth and elevation angle")


def p_data_type(
        input_type: typing.Any
) -> data_type.Type:
    if input_type == 'normal' or input_type == 1:
        return data_type.Type.normal
    elif input_type == 'depth' or input_type == 2:
        return data_type.Type.depth
    elif input_type == 'sketch' or input_type == 3:
        return data_type.Type.sketch
    elif input_type == 'silhouette' or input_type == 4:
        return data_type.Type.silhouette
    else:
        raise Exception("Given type should either be \"normal\" or \"depth\"!")
