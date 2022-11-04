from tick_config import *
from helpers import *


# Change this directory to specify data storage path
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

# Specify data directory under here if you stored data somewhere else in your machine
# Else just put the data in side the `data` directory
# e.g.: ./data/0050
# DATA_DIR = '/Users/anthonywah/Projects/mafs6100_order_placement/data'
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

CACHE_DIR = os.path.join(PROJECT_DIR, 'cache')

OPTRES_DIR = os.path.join(PROJECT_DIR, 'optres')


def get_one_stock_data(stock_code: str, verbose=True, gb_days=False) -> pd.DataFrame:
    """ Get all .csv.gz files from <DATA_DIR>/<stock_code>/*
        Will look for available cache first
        Save a cache file under cache/ as <stock_code>.pkl

    TODO from professor
    - Vary lambda and observe it's relationship with the change in score
    - DONE Investigate distribution of TAKE cases
    - DONE Include std heatmap plot as well
    - DONE Take square root on execution time in the objective to mimic formulation of brownian motion

    :param stock_code: a string of stock code, e.g. 0050
    :param verbose:
    :param gb_days:
    :return: dataframe
    """
    cache_path = os.path.join(CACHE_DIR, f'{stock_code}.pkl')
    st = datetime.datetime.now()
    if os.path.exists(cache_path):
        res = read_pkl_helper(cache_path)
        if gb_days:
            res = {k: v.reset_index(drop=True) for k, v in res.groupby('date')}
        if verbose:
            log_info(f'Got cache at {cache_path} - {(datetime.datetime.now() - st).total_seconds():.2f}s')

    else:
        sub_dir = os.path.join(DATA_DIR, stock_code)
        ym_ls = [i.split('.')[0].split('_')[-1] for i in os.listdir(sub_dir)]
        params_ls = [(stock_code, ym) for ym in ym_ls]
        res = pd.concat(pool_run_func(get_one_file, params_ls)).reset_index(drop=True)
        if verbose:
            log_info(f'{sub_dir}/* - Got {len(res)} entries from {len(ym_ls)} files - {(datetime.datetime.now() - st).total_seconds():.2f}s')
        save_pkl_helper(res, cache_path)
        if gb_days:
            res = {k: v.reset_index(drop=True) for k, v in res.groupby('date')}
        st = datetime.datetime.now()
        if verbose:
            log_info(f'Saved cache at {cache_path} - {(datetime.datetime.now() - st).total_seconds():.2f}s')
    return res


def save_by_date_cache(stock_code: str, df_dict: dict, verbose=True):
    """ Take in a df_dict grouped by date, and save them to cache/stock_code/<DATE>.pkl

    :param stock_code:
    :param df_dict:
    :param verbose:
    :return:
    """
    for d, df in df_dict.items():
        st = datetime.datetime.now()
        cache_path = os.path.join(CACHE_DIR, stock_code, f'{d}.pkl')
        if os.path.exists(cache_path):
            if verbose:
                log_info(f'Skipped existing cache {cache_path}')
        else:
            save_pkl_helper(df, cache_path)
            if verbose:
                log_info(f'Saved cache at {cache_path} - {(datetime.datetime.now() - st).total_seconds():.2f}s')
    return


def get_one_file(stock_code: str, year_month: str) -> pd.DataFrame:
    """ Get one .csv.gz file from <DATA_DIR>/<stock_code>/*

    :param stock_code: a string of stock code, e.g. 0050
    :param year_month: YYYYMM
    :return: dataframe
    """
    path = os.path.join(DATA_DIR, stock_code, f'{stock_code}_md_{year_month}_{year_month}.csv.gz')
    st = datetime.datetime.now()
    df = pd.read_csv(path, compression='gzip', index_col=0)
    df.loc[:, 'mid'] = (df['BP1'] + df['SP1']) / 2
    df.loc[:, 'w_mid'] = (df['BP1'] * df['SV1'] + df['SP1'] * df['BV1']) / (df['SV1'] + df['BV1'])
    df.loc[df['mid'] == 0, ['mid', 'w_mid']] = np.nan
    df.loc[:, 'mid'] = df['mid'].fillna(method='bfill')
    df.loc[:, 'w_mid'] = df['w_mid'].fillna(method='bfill')
    df.loc[:, 'spread'] = df['SP1'] - df['BP1']
    df.loc[:, 'spread_bps'] = 10000 * df['spread'] / df['mid']
    df.loc[:, 'tick_size'] = get_tick_size(bid_p=df['BP1'], ask_p=df['SP1'], sec_type=SEC_TYPE_DICT[stock_code])
    df.loc[:, 'spread_in_tick'] = (df['spread'] / df['tick_size']).round()
    df.loc[:, 'dt_time_str'] = df['time'].astype(str).str.rjust(9, '0')
    df.loc[:, 'dt_str'] = df['date'].str.replace('-', '') + df['dt_time_str']
    df.loc[:, 'dt'] = pd.to_datetime(df['dt_str'], format='%Y%m%d%H%M%S%f')
    df.loc[:, 'dt_str'] = df['dt'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df.loc[:, 'dt_time_str'] = df['dt'].dt.strftime('%H:%M:%S.%f')
    df.loc[:, 'dt_ms'] = (df['dt'].astype(int) / 1e6).round()
    df.loc[:, 'intv'] = df['dt_ms'].diff()
    df.loc[df['intv'] > 100000, 'intv'] = np.nan  # Day start row

    log_info(f'{path} - Got {len(df)} entries - {(datetime.datetime.now() - st).total_seconds():.2f}s')
    return df


def get_tick_size(bid_p: list, ask_p: list, sec_type: str) -> list:
    assert sec_type in TICK_CONFIG_LIST[0]['tick'].keys()
    df = pd.DataFrame({'bid': bid_p, 'ask': ask_p})
    df = (df / 100).round(3)  # original price are multiplied by 100
    df.loc[:, 'spread'] = df['ask'] - df['bid']
    df.loc[:, 'tick_size'] = 0
    min_bid, max_ask = df['bid'].min(), df['ask'].max()
    for lv in TICK_CONFIG_LIST:
        if min_bid >= lv['start']:
            df.loc[:, 'tick_size'] = lv['tick'][sec_type]
            continue
        elif max_ask < lv['start']:
            break
        df.loc[df['bid'] >= lv['start'], 'tick_size'] = lv['tick'][sec_type]
    df.loc[df['bid'] == 0, 'tick_size'] = np.nan  # trades that sweep multiple levels don't get orderbook update
    return (df['tick_size'] * 100).round()  # restore to 100 multiplier
