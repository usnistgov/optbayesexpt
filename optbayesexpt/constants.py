# check whether numba is available to load
# https://stackoverflow.com/questions/14050281/how-to-check-if-a-python-module-exists-without-importing-it

import sys, importlib
if sys.version_info[0] == 3:
    if sys.version_info[1] <= 3:
        if importlib.import_loader('numba') is not None:
            GOT_NUMBA = True
    if sys.version_info[1] >= 4:
        if importlib.util.find_spec('numba') is not None:
            GOT_NUMBA = True
else:
    GOT_NUMBA = False