from simulation import *
import seaborn as sns


def opt_main(stock_code, side, foms, ts_ls, tm_ls, overwrite=False):
    """ Main function to call for optimizing set of parameters on simulation

    :param stock_code:  target stock to simulate
    :param side:        target side to simulate
    :param foms:        target threshold for classifying flash orders
    :param ts_ls:       list of ts to optimize
    :param tm_ls:       list of tm to optimize
    :param overwrite:   whether to overwrite the results
    :return:
    """
    prefix = f'[{stock_code}-{side}-{foms}-{overwrite}]'
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
                log_info(f'{prefix.replace("]", f"-ts={ts}-tm={tm}]")} -- Skip invalid params --')
                continue
            sim_main(stock_code=stock_code, side=side, foms=foms, ts=ts, tm=tm, overwrite=overwrite)
    log_info(f'{prefix} Done optimization - {(datetime.datetime.now() - st).total_seconds():.2f}s')
    return


def get_path_df(update=False):
    """ Get a dataframe of path of simulation result, w.r.t corresponding parameters

    :param update: True if
    :return:
    """
    if not update:
        return read_pkl_helper(os.path.join(CACHE_DIR, 'path_df.pkl'))
    file_ls = glob.glob(os.path.join(PROJECT_DIR, 'optres', '**', '*.*'), recursive=True)
    file_split_ls = [(i, i.split('/')) for i in file_ls]
    path_df = pd.DataFrame([[f_ls[6], f_ls[7], f_ls[8], f_ls[9], f_ls[10], f] for f, f_ls in file_split_ls],
                           columns=['foms', 'stock_code', 'side', 'ts', 'tm', 'path'])
    for i in path_df.columns[:-1]:
        path_df.loc[:, i] = path_df[i].str.split('=', expand=True)[1]
    save_pkl_helper(path_df, os.path.join(CACHE_DIR, 'path_df.pkl'))
    return path_df


def get_sim_res(path_ls, cond=None, sequential=True):
    """ Take in a list of simulation result paths, get simulation results

    :param path_ls:     list of simulation results to be read
    :param cond:        function to apply condition to filter dataframe
    :param sequential:  true if read data sequentially but not in multi-processing
    :return:
    """

    full_res_path = [i for i in path_ls if 'full_res.pkl' in i]
    if len(full_res_path):
        df = read_pkl_helper(full_res_path[0])
    else:
        if sequential:
            df_ls = [read_csv_func(i) for i in path_ls]
        else:
            df_ls = pool_run_func(read_csv_func, path_ls)
        df = pd.concat(df_ls).reset_index(drop=True)
        save_pkl_helper(df, os.path.join(os.path.dirname(path_ls[0]), 'full_res.pkl'))

    # As we are adding CANCEL cases, need to filter out by pnl first
    df = df.loc[~df['pnl'].isna(), :].reset_index(drop=True)

    df.loc[:, 'duration'] = df['duration'].replace(0, np.nan)
    if cond:
        df = df.loc[cond(df), :].reset_index(drop=True)
    return {'pnl': df['pnl'].values, 'duration': df['duration'].values}


def plot_heatmap(stock_code_ls, side_ls, plot_attri_ls, res_dict, save_path=None):
    """ Plot the attributes in different heatmaps

    :param stock_code_ls:   list of stocks to plot
    :param side_ls:         list of target side to plot
    :param plot_attri_ls:   list of attributes to plot
    :param res_dict:        dict storing attributes of simulation results (mean score, take % ... etc)
    :param save_path:       Path to save the figure, if save_path is None then plt.show() will be called
    :return:
    """
    col_len, row_len = len(stock_code_ls), len(side_ls) * len(plot_attri_ls)
    fig, axs = plt.subplots(figsize=(26 * col_len, 12 * row_len), ncols=col_len, nrows=row_len)
    for i_sc, stock_code in enumerate(stock_code_ls):
        for i_s, side in enumerate(side_ls):
            for i_p, plot_attri in enumerate(plot_attri_ls):
                log_info(f'Plotting {stock_code} - {side} - {plot_attri}')
                if len(stock_code_ls) > 1:
                    ax = axs[(i_s * len(plot_attri_ls)) + i_p, i_sc]
                elif len(side_ls) * len(plot_attri_ls) > 1:
                    ax = axs[(i_s * len(plot_attri_ls)) + i_p]
                else:
                    ax = axs
                target_res_dict = {k: v for k, v in res_dict.items() if stock_code in k and side in k}
                ts_ls = sorted(list(set([int(i[3]) for i in target_res_dict.keys()])))
                tm_ls = sorted(list(set([int(i[4]) for i in target_res_dict.keys()])))
                matrix_df = pd.DataFrame(np.nan, index=ts_ls, columns=tm_ls)
                for k, v in target_res_dict.items():
                    matrix_df.loc[int(k[3]), int(k[4])] = v[plot_attri]
                sns.heatmap(matrix_df.loc[ts_ls, tm_ls], ax=ax, annot=True, fmt='.2g', annot_kws={"fontsize": 8})
                ax.set_xlabel('tm', fontsize=12)
                ax.set_ylabel('ts', fontsize=12)
                ax.set_title(f'{stock_code} - {side} - {plot_attri}', fontsize=16)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    return


def calc_sim_results(path_df, obj_kwargs=None, read_cache=False):
    """ A wrapped runner for calculating results of simulations given list of paths to run (as a dataframe)

    :param path_df:     dataframe listing out paths of simulation results to read
    :param obj_kwargs:  kwargs for objective function
    :param read_cache:  opt-in for reading saved result
    :return:
    """
    cache_path = os.path.join(CACHE_DIR, 'calc_sim_res.pkl')
    if read_cache and os.path.exists(cache_path):
        return read_pkl_helper(os.path.join(CACHE_DIR, 'calc_sim_res.pkl'))
    gb = path_df.groupby(['stock_code', 'foms', 'side', 'ts', 'tm'])
    if obj_kwargs is None:
        obj_kwargs = {'lmda': 0, 't_func': lambda x: np.sqrt(x / 1000.)}
    res = {}
    for gp_key, df in tqdm.tqdm(gb, desc='GettingSimData', ncols=200, total=len(gb.groups)):
        if gp_key not in res.keys():
            res[gp_key] = get_sim_res(path_ls=df['path'].unique().tolist())

        # EoQ cases
        if 'eoq_res' not in res[gp_key].keys():
            res[gp_key]['eoq_res'] = get_sim_res(path_ls=df['path'].unique().tolist(), cond=lambda xdf: xdf['eoq_rep'])
        eoq_score = obj2(res[gp_key]['eoq_res']['pnl'], res[gp_key]['eoq_res']['duration'], **obj_kwargs)
        eoq_score = eoq_score[np.logical_and(~np.isnan(eoq_score), ~np.isinf(eoq_score))]
        res[gp_key].update({
            'eoq_scores': eoq_score,
            'eoq_count_k': len(eoq_score) / 1000,
            'eoq_score_mean': np.nanmean(eoq_score),
            'eoq_score_std': np.nanstd(eoq_score)
        })

        # Take cases
        if 'take_res' not in res[gp_key].keys():
            res[gp_key]['take_res'] = get_sim_res(path_ls=df['path'].unique().tolist(), cond=lambda xdf: xdf['case'] == 'TAKE')
        take_score = obj2(res[gp_key]['take_res']['pnl'], res[gp_key]['take_res']['duration'], **obj_kwargs)
        take_score = take_score[np.logical_and(~np.isnan(take_score), ~np.isinf(take_score))]
        res[gp_key].update({
            'take_scores': take_score,
            'take_count_k': len(take_score) / 1000,
            'take_score_mean': np.nanmean(take_score),
            'take_score_std': np.nanstd(take_score)
        })

        # Calculate score
        obj_kwargs = {'lmda': 0, 't_func': lambda x: np.sqrt(x / 1000.)}
        res_score = obj2(res[gp_key]['pnl'], res[gp_key]['duration'], **obj_kwargs)
        res_score = res_score[np.logical_and(~np.isnan(res_score), ~np.isinf(res_score))]
        res[gp_key].update({
            'scores': res_score,
            'count_k': len(res_score) / 1000,
            'score_mean': np.nanmean(res_score),
            'score_std': np.nanstd(res_score),
            'eoq_%': len(res[gp_key]['eoq_scores']) / len(res_score),
            'take_%': len(res[gp_key]['take_scores']) / len(res_score),
        })

        for i in ['eoq_', 'take_', '']:
            res[gp_key][f'{i}z_score'] = (res[gp_key][f'{i}score_mean'] + 30) / res[gp_key][f'{i}score_std']

    log_info(f'Got simulation results on {len(res)} set of params')
    if read_cache:  # Save for next time use
        save_pkl_helper(res, save_path=cache_path)
    return res


if __name__ == '__main__':
    """
    T star needs to > 0
    T mid needs to be >= 0 and <= T star 
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-C', '--code', dest='stock_code', type=str, default='', help='Stock code to simulate')
    arg_parser.add_argument('-S', '--side', dest='side', type=str, default='', help='bid or ask')
    arg_parser.add_argument('-F', '--foms', dest='foms', type=int, default='', help='flash order ms')
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
        foms=args.foms,
        ts_ls=list(range(args.ts_start, args.ts_end + 1, args.ts_step)),
        tm_ls=list(range(args.tm_start, args.tm_end + 1, args.tm_step)),
        overwrite=args.overwrite
    )



