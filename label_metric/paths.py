import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')

DATA_DIR = '/data/scratch/acw751'
OrchideaSOL_DIR = os.path.join(DATA_DIR, '_OrchideaSOL2020_release')


if __name__ == '__main__':
    print(ROOT_DIR)
    print(CONFIGS_DIR)
    print(DATA_DIR)
    print(OrchideaSOL_DIR)
    