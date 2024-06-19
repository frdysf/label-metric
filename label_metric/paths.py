import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')

if __name__ == '__main__':
    print(ROOT_DIR)
    print(CONFIGS_DIR)
