import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
DATA_DIR_EECS = None
DATA_DIR_APOCRITA = '/data/home/acw745/datasets/_OrchideaSOL2020_release'

if __name__ == '__main__':
    print(ROOT_DIR)
    print(CONFIGS_DIR)
