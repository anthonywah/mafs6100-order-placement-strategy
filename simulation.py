import matplotlib.pyplot as plt
from get_data import *
import tqdm
import glob
import datetime
from argparse import ArgumentParser


PLOT_CASE_LABEL = {
    'REP': 'Replaced Order Fill',
    'INIT': 'Initial Order Fill',
    'TAKE': 'Timeout Take'
}


def obj(pnl, t_exec, lmda=1, t_func=lambda x: x):
    """ Objective function for measuring the spread narrowing strategy - score = (lmda / t_func(t_exec)) + pnl

    :param pnl: pnl against prevailing mid quote, in bps
    :param t_exec: execution time in millisecond
    :param lmda: Scaling factor
    :param t_func: Transformation function on t_exec
    :return:
    """
    return (lmda / t_func(t_exec)) + pnl


def sim_one_day_t2(date: str, stock_code:str, side: str, ts: int, tm: int, verbose=False, save=False, overwrite=False) -> pd.DataFrame:
    """ Simulate cases where we post an order when spread in tick = 2

    :param date: date to simulate on
    :param stock_code: stock code to simulate
    :param side: 'bid' or 'ask'
    :param ts: time to wait til crossing spread, in seconds
    :param tm: time to wait til replacing order with 1 tick better price, in seconds
    :param verbose: print computing progress or not
    :param save: If true save result to optres
    :return:
    """
    st = datetime.datetime.now()
    cache_path = os.path.join(CACHE_DIR, stock_code, f'{date}.pkl')
    try:
        df = read_pkl_helper(cache_path)
    except FileNotFoundError as e:
        log_error(f'Cache does not exist: {cache_path}')
        raise e
    save_path = os.path.join(OPTRES_DIR, f'stock_code={stock_code}', f'side={side}', f'ts={ts}', f'tm={tm}', f'{date}.csv')
    if save and os.path.exists(save_path) and not overwrite:
        return
    ts_ms, tm_ms = int(ts * 1e3), int(tm * 1e3)
    res = []
    side_coef = -1 if side == 'bid' else 1
    make_p_col = 'BP1' if side == 'bid' else 'SP1'
    take_p_col = 'SP1' if side == 'bid' else 'BP1'
    prev_take_p, prev_dur = None, None
    col_ls = ['date', 'make_price', 'fill_price', 'case', 'duration', 'pnl', 'start_ts', 'start_index', 'start_price',
              'start_bid', 'start_ask', 'replace_ts', 'replace_index', 'replace_p', 'replace_bid', 'replace_ask',
              'fill_ts', 'fill_index', 'fill_bid', 'fill_ask']

    # Reset variables for one order
    s_ms, s_bid, s_ask, r_ms, r_bid, r_ask, f_ms, f_bid, f_ask, s_mid = 0, 0., 0., 0, 0., 0., 0, 0., 0., 0.
    w_r, fill, rep, dur, case, s_p, m_p, r_p, f_p, pnl, s_i, r_i, f_i = False, False, False, 0, '', 0., 0., 0., 0., 0., -1, -1, -1

    iter_obj = df.iterrows() if not verbose else tqdm.tqdm(df.iterrows(), desc=f'{date}:Simulation', ncols=200, total=len(df))
    for i, i_row in iter_obj:

        # Skip multi-level trades updates
        if i_row['BP1'] == 0:
            continue

        # Handle ongoing round
        if s_ms > 0:
            dur = i_row['dt_ms'] - s_ms
            to_rep_p = m_p + (side_coef * -1) * i_row['tick_size']

            if dur >= ts_ms:
                # 1. Time exceed t_star and cross spread
                dur = ts_ms
                f_p, f_ms = i_row[take_p_col], s_ms + ts_ms
                case, fill = 'TAKE', True

            elif rep:
                if (side_coef * (i_row['lastPx'] - m_p)) >= 0 or (side_coef * (i_row[take_p_col] - m_p)) >= 0:
                    # 2. Replaced order filled by either 1 tick narrower of opposite side or trade update at my price
                    f_p, f_ms = m_p, i_row['dt_ms']
                    case, fill = 'REP', True

            elif side_coef * (m_p - i_row[make_p_col]) > 0:
                # Cancel order (take precedence over replacing order after waiting to erase uncertainty)
                s_ms, s_bid, s_ask, r_ms, r_bid, r_ask, f_ms, f_bid, f_ask, s_mid = 0, 0., 0., 0, 0., 0., 0, 0., 0., 0.
                w_r, fill, rep, dur, case, s_p, m_p, r_p, f_p, pnl, s_i, r_i, f_i = False, False, False, 0, '', 0., 0., 0., 0., 0., -1, -1, -1

            elif dur >= tm_ms:
                # Replace order
                if (not to_rep_p == i_row[take_p_col]) or (to_rep_p != prev_take_p and prev_dur < tm_ms):
                    m_p = to_rep_p
                    r_ms, r_bid, r_ask, r_p = s_ms + tm_ms, i_row['BP1'], i_row['SP1'], m_p
                    r_i, rep = i, True
                    if w_r:
                        r_ms = i_row['dt_ms']
                else:
                    w_r = True

            elif (side_coef * (i_row['lastPx'] - m_p)) > 0 or (side_coef * (i_row[take_p_col] - m_p)) >= 0:
                # 3. Non-rep order filled by either 1 tick narrower of opposite side or trade update at my price
                f_p, f_ms = m_p, i_row['dt_ms']
                case, fill = 'INIT', True

            # Append result and reset
            if fill:
                f_i, f_bid, f_ask, pnl = i, i_row['BP1'], i_row['SP1'], side_coef * 10000 * (f_p - s_mid) / s_mid
                res.append([date, m_p, f_p, case, dur, pnl, s_ms, s_i, s_p, s_bid, s_ask, r_ms, r_i, r_p, r_bid, r_ask, f_ms, f_i, f_bid, f_ask])
                s_ms, s_bid, s_ask, r_ms, r_bid, r_ask, f_ms, f_bid, f_ask, s_mid = 0, 0., 0., 0, 0., 0., 0, 0., 0., 0.
                w_r, fill, rep, dur, case, s_p, m_p, r_p, f_p, pnl, s_i, r_i, f_i = False, False, False, 0, '', 0., 0., 0., 0., 0., -1, -1, -1

        # No ongoing round - check if need to start a new round
        elif i_row['spread_in_tick'] == 2:
            s_ms, s_bid, s_ask, s_p = i_row['dt_ms'], i_row['BP1'], i_row['SP1'], i_row[make_p_col]
            s_i, m_p, s_mid = i, i_row[make_p_col], i_row['mid']

        # buffer for new updates
        prev_take_p, prev_dur = i_row[take_p_col], dur

    res_df = pd.DataFrame(res, columns=col_ls)
    if not save:
        time_used = (datetime.datetime.now() - st).total_seconds()
        if verbose:
            log_info(f'Done Simulation on {date} - {len(res_df)} results from {len(df)} quotes - {time_used:.3f}s')
        return res_df
    os.makedirs(os.path.dirname(save_path), mode=0o755, exist_ok=True)
    res_df.to_csv(save_path, index=False)
    time_used = (datetime.datetime.now() - st).total_seconds()
    if verbose:
        log_info(f'Saved at {save_path} - {time_used:.3f}s')
    return


def plot_one_sim(day_df: pd.DataFrame, ts: int, tm: int, side: str, row: pd.Series):
    """ Plot graph of one experiment result

    :param day_df: price dataframe of simulation day
    :param ts: t_star
    :param tm: t_m
    :param side: 'bid' or 'ask'
    :param row: pandas series storing one simulation result
    :return:
    """
    date, m_p, f_p, case, dur, pnl, s_ms, s_i, s_p, s_bid, s_ask, r_ms, r_i, r_p, r_bid, r_ask, f_ms, f_i, f_bid, f_ask = row
    sdf = day_df.loc[max(s_i - 10, 0):min(f_i + 10, day_df.shape[0] - 1), :].copy()
    sdf = sdf[[i for i in sdf.columns if not (i[0] in ['B', 'S'] and i not in ['BP1', 'SP1'])]]
    sdf.loc[:, 'order_price'] = np.nan
    sdf.loc[s_i, 'order_price'] = s_p
    if r_i > 0:
        if r_ms != sdf.loc[r_i, 'dt_ms']:
            sdf = insert_new_row(orig_df=sdf, new_ms=r_ms, new_i=r_i, new_p=r_p)
            f_i += 1
        else:
            sdf.loc[r_i, 'order_price'] = r_p
    sdf.loc[:, 'order_price'] = sdf['order_price'].fillna(method='ffill')
    if f_ms != sdf.loc[f_i, 'dt_ms']:
        sdf = insert_new_row(orig_df=sdf, new_ms=f_ms, new_i=f_i, new_p=f_p)
    sdf.loc[f_i:, 'order_price'] = np.nan
    sdf.loc[:, 't'] = sdf['dt_ms'] - s_ms
    sdf = sdf[[i for i in sdf.columns if not (i[0] in ['B', 'S'] and i not in ['BP1', 'SP1'])]]
    for p_col in ['SP1', 'BP1']:
        sdf.loc[:, p_col] = sdf[p_col].replace(0, np.nan).fillna(method='ffill')

    # Plot starts
    fig, ax = plt.subplots(figsize=(16, 8))
    t_df = sdf[['t', 'lastPx']].dropna()
    s_t = sdf.loc[s_i, 't']
    ax.plot(sdf['t'], sdf['BP1'], color='black', alpha=0.6, drawstyle='steps-post')
    ax.plot(sdf['t'], sdf['SP1'], color='black', alpha=0.6, drawstyle='steps-post')
    ax.plot(sdf['t'], sdf['order_price'], color='green', label='Order Price', drawstyle='steps-post', linewidth=3.0)
    ax.scatter(t_df['t'], t_df['lastPx'], color='blue', label='Market Trades', alpha=0.7, s=80, marker='D')
    ax.scatter(f_ms - s_ms + s_t, f_p, s=120, marker='<', color='green', label=PLOT_CASE_LABEL[case])
    ax.axvline(s_t, color='violet')
    if tm * 1e3 + s_t <= sdf['t'].max():
        ax.axvline(tm * 1e3 + s_t, color='violet')
    if ts * 1e3 + s_t <= sdf['t'].max():
        ax.axvline(ts * 1e3 + s_t, color='violet')
    ax.axhline((s_bid + s_ask) / 2, linestyle='--', color='red', alpha=0.5, label='Prevailing Mid')
    ax.grid(True, alpha=0.5)
    ax.set_title(f'Side={side}; Case={case}; Duration={dur:.2f}ms; Make={m_p:.2f}; Fill={f_p:.2f}; PnL={pnl:.2f}bps', fontsize=16)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return sdf


def insert_new_row(orig_df, new_ms, new_i, new_p):
    """ Since replace / take may happen in between orderbook updates, insert the action row to the dataframe

    :param orig_df:
    :param new_ms:
    :param new_i:
    :param new_p:
    :return:
    """
    old_index_start = orig_df.index.tolist()[0]
    f_dt = pd.to_datetime(new_ms * 1e6)
    d = orig_df.loc[new_i].to_dict()
    d.update({
        'lastPx': np.nan, 'size': np.nan, 'dt_ms': new_ms, 'dt': f_dt, 'order_price': new_p,
        'dt_str': f_dt.strftime('%Y-%m-%d %H:%M:%S.%f'), 'dt_time_str': f_dt.strftime('%H:%M:%S.%f'),
        'time': int(f_dt.strftime('%-H%M%S%f')[:-3])
    })
    d_df = pd.DataFrame(d, index=[new_i])
    df = pd.concat([orig_df.loc[:(new_i - 1)], d_df, orig_df.loc[new_i:]]).reset_index(drop=True)
    df.index += old_index_start
    return df


def sim_main(stock_code, side, ts, tm, overwrite):
    """ Main function to simulate experiment on one set of params

    :param stock_code:
    :param side:
    :param ts:
    :param tm:
    :param overwrite:
    :return:
    """
    st = datetime.datetime.now()
    prefix = f'[{stock_code}-{side}-ts={ts}-tm={tm}]'
    path_ls = glob.glob(os.path.join(PROJECT_DIR, 'optres', f'stock_code={stock_code}', f'side={side}', f'ts={ts}', f'tm={tm}', '*'))
    if overwrite:
        for path in path_ls:
            os.remove(path)
    else:
        if len(path_ls):
            log_info(f'{prefix} Simulation result exists')
            return
    date_ls = [i.replace('.pkl', '') for i in os.listdir(os.path.join(CACHE_DIR, stock_code))]
    params_ls = [[d, stock_code, side, ts, tm, False, True, overwrite] for d in date_ls]
    pool_run_func(sim_one_day_t2, params_ls)
    log_info(f'{prefix} Done simulation on {len(params_ls)} days - {(datetime.datetime.now() - st).total_seconds():.2f}s')
    return


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-C', '--code', dest='stock_code', type=str, default='', help='Stock code to simulate')
    arg_parser.add_argument('-S', '--side', dest='side', type=str, default='', help='bid or ask')
    arg_parser.add_argument('-s', '--ts', dest='ts', type=int, default=-1, help='T star in seconds')
    arg_parser.add_argument('-m', '--tm', dest='tm', type=int, default=-1, help='T mid in seconds')
    arg_parser.add_argument('-o', '--overwrite', dest='overwrite', type=bool, default=False, help='Overwrite or not')
    args = arg_parser.parse_args()

    assert args.stock_code in os.listdir(DATA_DIR)
    assert args.side in ('ask', 'bid')
    assert args.ts > 0 and 0 <= args.tm <= args.ts

    sim_main(stock_code=args.stock_code, side=args.side, ts=args.ts, tm=args.tm, overwrite=args.overwrite)
