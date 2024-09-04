import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
DATA_DIR_EECS = '/homes/ht156/data/_OrchideaSOL2020_release'
DATA_DIR_APOCRITA = '/data/scratch/acw751/_OrchideaSOL2020_release'

if __name__ == '__main__':
    print(ROOT_DIR)
    print(CONFIGS_DIR)
