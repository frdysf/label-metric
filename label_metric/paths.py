import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = '/data/scratch/acw751'
OrchideaSOL_DIR = os.path.join(DATA_DIR, '_OrchideaSOL2020_release')

# vocalset_dir = os.path.join(scratch_dir, 'VocalSet')

if __name__ == '__main__':
    print(ROOT_DIR)
    print(DATA_DIR)
    print(OrchideaSOL_DIR)