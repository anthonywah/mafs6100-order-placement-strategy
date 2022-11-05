from simulation import *
import seaborn as sns


def opt_main(stock_code, side, ts_ls, tm_ls, overwrite=False):
    prefix = f'[{stock_code}-{side}-{overwrite}]'
    log_info(f'{prefix} Start optimization on {len(ts_ls) * len(tm_ls)} param sets')
    st = datetime.datetime.now()
    cache_dir = os.path.join(CACHE_DIR, stock_code)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        df_dict = get_one_stock_data(stock_code=stock_code, verbose=False, gb_days=True)
        save_by_date_cache(stock_code=stock_code, df_dict=df_dict, verbose=True)
    for ts in ts_ls:
        for tm in tm_ls:
            if not (ts > 0 and ts >= tm >= 0):
                log_info(f'[{stock_code}-{side}-ts={ts}-tm={tm}] -- Skip invalid params --')
                continue
            sim_main(stock_code=stock_code, side=side, ts=ts, tm=tm, overwrite=overwrite)
    log_info(f'{prefix} Done optimization - {(datetime.datetime.now() - st).total_seconds():.2f}s')
    return


def get_path_df(stock_code, side):
    """ Get a dataframe of path of simulation result, w.r.t corresponding parameters

    :param stock_code:
    :param side:
    :return:
    """
    file_ls = glob.glob(os.path.join(PROJECT_DIR, 'optres', f'stock_code={stock_code}', f'side={side}', '**', '*.*'),
                        recursive=True)
    file_split_ls = [(i, i.split('/')) for i in file_ls]
    files_df = pd.DataFrame([[f_ls[6], f_ls[7], f_ls[8], f_ls[9], f] for f, f_ls in file_split_ls],
                            columns=['stock_code', 'side', 'ts', 'tm', 'path'])
    for i in files_df.columns[:-1]:
        files_df.loc[:, i] = files_df[i].str.split('=', expand=True)[1]
    return files_df


def get_sim_res(path_ls, cond=None):
    """ Take in a list of simulation result paths, get simulation results

    :param path_ls:
    :param cond: function to apply condition to filter dataframe
    :return:
    """

    full_res_path = [i for i in path_ls if 'full_res.pkl' in i]
    if len(full_res_path):
        df = read_pkl_helper(full_res_path[0])
    else:
        df_ls = pool_run_func(read_csv_func, path_ls)
        df = pd.concat(df_ls).reset_index(drop=True)
        save_pkl_helper(df, os.path.join(os.path.dirname(path_ls[0]), 'full_res.pkl'))
    df.loc[:, 'duration'] = df['duration'].replace(0, np.nan)
    if cond:
        df = df.loc[cond(df), :].reset_index(drop=True)
    return {'pnl': df['pnl'].values, 'duration': df['duration'].values}


def plot_heatmap(stock_code, side, plot_attri, res_dict):
    target_res_dict = {k: v for k, v in res_dict.items() if stock_code in k and side in k}
    ts_ls = sorted(list(set([int(i[2]) for i in target_res_dict.keys()])))
    tm_ls = sorted(list(set([int(i[3]) for i in target_res_dict.keys()])))
    matrix_df = pd.DataFrame(np.nan, index=[f'ts={i}' for i in ts_ls], columns=[f'tm={i}' for i in tm_ls])
    for k, v in target_res_dict.items():
        matrix_df.loc[f'ts={k[2]}', f'tm={k[3]}'] = v[plot_attri]
    fig, ax = plt.subplots(figsize=(22, 10))
    sns.heatmap(matrix_df, ax=ax, annot=True, fmt='.2g', annot_kws={"fontsize": 8})
    ax.set_title(f'{stock_code} - {side} - {plot_attri}')
    plt.show()
    return


if __name__ == '__main__':
    """
    T star needs to > 0
    T mid needs to be >= 0 and <= T star 
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-C', '--code', dest='stock_code', type=str, default='', help='Stock code to simulate')
    arg_parser.add_argument('-S', '--side', dest='side', type=str, default='', help='bid or ask')
    arg_parser.add_argument('-o', '--overwrite', dest='overwrite', type=bool, default=False, help='Overwrite or not')
    arg_parser.add_argument('--ts_start', dest='ts_start', type=int, default=-1, help='T star start')
    arg_parser.add_argument('--ts_step', dest='ts_step', type=int, default=-1, help='T star step')
    arg_parser.add_argument('--ts_end', dest='ts_end', type=int, default=-1, help='T star end')
    arg_parser.add_argument('--tm_start', dest='tm_start', type=int, default=1, help='T mid start')
    arg_parser.add_argument('--tm_step', dest='tm_step', type=int, default=1, help='T mid step')
    arg_parser.add_argument('--tm_end', dest='tm_end', type=int, default=1, help='T mid end')
    args = arg_parser.parse_args()

    assert args.stock_code in os.listdir(DATA_DIR)
    assert args.side in ('ask', 'bid')
    assert 0 < args.ts_start < args.ts_end and args.ts_step > 0
    assert 0 < args.tm_start < args.tm_end and args.tm_step > 0
    opt_main(
        stock_code=args.stock_code,
        side=args.side,
        ts_ls=list(range(args.ts_start, args.ts_end + 1, args.ts_step)),
        tm_ls=list(range(args.tm_start, args.tm_end + 1, args.tm_step)),
        overwrite=args.overwrite
    )



