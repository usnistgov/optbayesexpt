import sys, importlib
version = '1.2.0'

# check whether numba is available to load
# https://stackoverflow.com/questions/14050281/how-to-check-if-a-python-module-exists-without-importing-it
GOT_NUMBA = False
if sys.version_info[0] == 3:
    if sys.version_info[1] <= 3:
        numba_module = importlib.import_loader('numba')
    if sys.version_info[1] >= 4:
        numba_module = importlib.util.find_spec('numba')

    if numba_module is not None:
        GOT_NUMBA = True

if __name__ == '__main__':
    print(f'version = {version}')
    print(f'GOT_NUMBA = {GOT_NUMBA}')
    if GOT_NUMBA:
        print(numba_module)