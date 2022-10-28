import datetime
import os
import _pickle as cp
import json
import pandas as pd
import logging
import numpy as np
from pytz import timezone, utc
import multiprocessing as mp


FORMAT_STR = "%(asctime)s.%(msecs)03d | {}%(levelname)-8s | %(filename)-20s:%(lineno)5s{} | %(message)s"


def custom_tz(*args):
    utc_dt = utc.localize(datetime.datetime.utcnow())
    my_tz = timezone('Asia/Hong_Kong')
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


class MyFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.format_str = FORMAT_STR.format('', '')
        self.date_fmt = '%Y-%m-%d %H:%M:%S'

    def format(self, record):
        log_fmt = self.format_str
        formatter = logging.Formatter(log_fmt, self.date_fmt)
        return formatter.format(record)


def get_logger(name):
    logger = logging.getLogger(name)
    logging.Formatter.converter = custom_tz
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(MyFormatter())
    logger.addHandler(ch)
    logger.propagate = False
    return logger


helper_logger = get_logger('helper')
log_info = helper_logger.info
log_error = helper_logger.error


def read_csv_func(read_path, **kwargs):
    return pd.read_csv(read_path, **kwargs)


def save_csv_func(save_df, save_path, **kwargs):
    save_df.to_csv(save_path, **kwargs)
    return


def read_json_func(read_path, **kwargs):
    with open(read_path, 'r') as f:
        res_dict = json.load(f, **kwargs)
    return res_dict


def save_json_func(save_json, save_path, **kwargs):
    with open(save_path, 'w') as f:
        json.dump(save_json, **kwargs)
    return


def read_pkl_helper(read_path, **kwargs):
    with open(read_path, 'rb') as f:
        res = cp.load(f, **kwargs)
    return res


def save_pkl_helper(save_pkl, save_path, **kwargs):
    with open(save_path, 'wb') as f:
        cp.dump(save_pkl, f, **kwargs)
    return


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def print_dict(input_dict):
    print(json.dumps(input_dict, indent=4, cls=NpEncoder))


def get_cur_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def pool_run_func(func, arg_ls):
    if type(arg_ls[0]) not in [list, tuple]:
        arg_ls = [[i] for i in arg_ls]
    with mp.Pool(os.cpu_count() * 2) as p:
        res = p.starmap(func=func, iterable=arg_ls)
        p.terminate()
    return res


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

