def parse(string):
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        raise Exception("Given value was neither \"True\" nor \"False\"")
