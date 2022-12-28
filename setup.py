import os
try:  # Try to install required packages first
    cmd = f'pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt'
    print(f'running {cmd}')
    os.system(cmd)
except Exception as e:
    pass
from optimization import *
from flash_order import *

""" Start up Script to ensure one has appropriate paths to run other scripts """

if __name__ == '__main__':

    # Ensure data exists
    for i in ['0050', '2330']:
        assert i in os.listdir(DATA_DIR)
        assert i in os.listdir(CACHE_DIR)

    # Ensure cache files exist
    for code in ['0050', '2330']:
        df_gb = get_one_stock_data(stock_code=code, verbose=True, gb_days=True)

    # Ensure optres files exist
    for i in [0, 300]:
        assert f'foms={i}' in os.listdir(OPTRES_DIR)

    # Ensure paths in cache are correct
    path_df = get_path_df(update=True)

    log_info('Repo setup ready')
