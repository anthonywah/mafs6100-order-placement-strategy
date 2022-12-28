from optimization import *
from flash_order import *
import warnings
warnings.filterwarnings('ignore')

# Serve as a cache for all the plots
log_info(f'Preparing cache data for generating figures ... ')

target_stock_code = '0050'

path_df = get_path_df(update=False)

gb = path_df.groupby(['stock_code', 'foms', 'side', 'ts', 'tm'])

res_dict = calc_sim_results(path_df=path_df, read_cache=True)

stock_df = get_one_stock_data(stock_code=target_stock_code)

stock_df_dict = {k: v.reset_index(drop=True) for k, v in stock_df.groupby('date')}

samp_sim_df = sim_one_day_t2('2022-01-03', target_stock_code, 'bid', 10, 5, 0, save=False, verbose=False)

samp_foc = FlashOrderCalculator(target_stock_code)

samp_foc.classify(max_dur_ms=100, trick_trade_thres_ms=100)

log_info(f'Cache data all ready')


def get_figure_path(fig_num):
    return os.path.join(PROJECT_DIR, 'report', 'fig', f'figure_{fig_num}.png')


def figure_1():
    """ Stage 1 """
    save_path = get_figure_path(1)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title('Spread (in tick)')
    ax.hist(stock_df['spread_in_tick'], bins=200)
    ax.grid(True)
    fig.savefig(save_path)
    return


def figure_2():
    """ Stage 2"""
    save_path = get_figure_path(2)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    all_df = stock_df.reset_index(drop=True)
    all_df.loc[:, '5m_bin'] = all_df['dt'].dt.floor(freq='5min').dt.time.astype(str)
    df_bin = all_df.groupby('5m_bin')[['spread_bps']].mean().reset_index()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title('Spread Profile (in bps)')
    ax.plot(df_bin['5m_bin'], df_bin['spread_bps'], label='5-minute bin')
    ax.plot(df_bin['5m_bin'], df_bin['spread_bps'].rolling(10).mean(), color='red', alpha=0.3, label='10MA')
    locator = MaxNLocator(prune='both', nbins=8)
    ax.xaxis.set_major_locator(locator)
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path)
    return


def figure_3():
    """ Stage """
    save_path = get_figure_path(3)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    row = samp_sim_df.loc[627]
    plot_one_sim(day_df=stock_df_dict.get('2022-01-03').copy(), ts=10, tm=5, side='bid', row=row, save_path=save_path)
    return


def figure_4():
    """ Stage """
    save_path = get_figure_path(4)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    row = samp_sim_df.loc[379]
    plot_one_sim(day_df=stock_df_dict.get('2022-01-03').copy(), ts=10, tm=5, side='bid', row=row, save_path=save_path)
    return


def figure_5():
    """ Stage """
    save_path = get_figure_path(5)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    one_dict = {k: v for k, v in res_dict.items() if k[0] == '0050' and k[1] == '0' and k[2] == 'bid' and max(int(k[3]), int(k[4])) <= 20}
    plot_heatmap(stock_code_ls=['0050'], side_ls=['bid'], plot_attri_ls=['score_mean', 'score_std'], res_dict=one_dict, save_path=save_path)
    return


def figure_6():
    """ Stage """
    save_path = get_figure_path(6)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    bid_dur = []
    ask_dur = []
    for item in samp_foc.fod[100][100]:
        if item['side'] == 'bid':
            bid_dur.append(item['duration'])
        elif item['side'] == 'ask':
            ask_dur.append(item['duration'])
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hist(bid_dur, bins=200, label=f'bid quotes, mean = {np.mean(bid_dur)}, std = {np.std(bid_dur)}')
    ax.hist(ask_dur, bins=200, label=f'ask quotes, mean = {np.mean(ask_dur)}, std = {np.std(ask_dur)}', alpha=0.6)
    ax.set_title('bid/ask duration distribution')
    ax.legend(loc='upper right')
    fig.savefig(save_path)
    return


def figure_7():
    """ Stage """
    save_path = get_figure_path(7)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    bid_influ, ask_influ = [], []
    for i in range(1, 31):
        bid_mv, ask_mv = samp_foc.get_flick_influ(fo_dur=100, decay=i)
        bid_influ.append(bid_mv)
        ask_influ.append(ask_mv)

    fig, ax = plt.subplots(figsize=(16, 8))
    x_bid = np.array([np.mean(bid_mv) for bid_mv in bid_influ])
    s_bid = np.array([np.std(bid_mv) for bid_mv in bid_influ])
    ax.plot(np.linspace(1, 31, 30), x_bid, label='bid flick')
    ax.fill_between(np.linspace(1, 31, 30), x_bid - s_bid, x_bid + s_bid, alpha=0.3)
    x_ask = np.array([np.mean(ask_mv) for ask_mv in ask_influ])
    s_ask = np.array([np.std(ask_mv) for ask_mv in ask_influ])
    ax.plot(np.linspace(1, 31, 30), x_ask, label='ask flick')
    ax.fill_between(np.linspace(1, 31, 30), x_ask - s_ask, x_ask + s_ask, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Term structure of Mean of return after bid / ask flickering quotes')
    ax.set_xlabel(f'seconds after flickering quotes of {target_stock_code}')
    ax.set_ylabel('mean of return')
    fig.savefig(save_path)
    return


def figure_8():
    """ Stage 5: Heatmap of simulations of 0050, bid, tm=1-20, ts=1-20, with consideration of fo, foms=300 """
    save_path = get_figure_path(8)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    one_dict = {k: v for k, v in res_dict.items() if k[0] == target_stock_code and k[1] == '300' and k[2] == 'bid'}
    plot_heatmap(stock_code_ls=['0050'], side_ls=['bid'], plot_attri_ls=['score_mean'], res_dict=one_dict, save_path=save_path)
    return


def figure_9():
    """ Stage 5: Cases distribution breakdown represented by a bar chart """
    save_path = get_figure_path(9)
    log_info(f'Plotting {os.path.basename(save_path).split(".")[0]}')
    stock_code, side, ts, tm, foms = target_stock_code, 'bid', 20, 10, 300
    w_fo_path = os.path.join(PROJECT_DIR, f'optres/foms={foms}/stock_code={stock_code}/side={side}/ts={ts}/tm={tm}', 'full_res.pkl')
    wo_fo_path = w_fo_path.replace('foms=300', 'foms=0')
    sim_res_w_fo, sim_res_wo_fo = read_pkl_helper(w_fo_path), read_pkl_helper(wo_fo_path)
    w_fo_gb, wo_fo_gb = sim_res_w_fo.groupby('case'), sim_res_wo_fo.groupby('case')
    w_fo_cnt, wo_fo_cnt = sim_res_w_fo.shape[0], sim_res_wo_fo.shape[0]
    w_fo_xy, wo_fo_xy = [], []
    for i, k in enumerate(PLOT_CASE_LABEL.keys()):
        k_w_fo_cnt, k_wo_fo_cnt = len(w_fo_gb.groups.get(k, [])), len(wo_fo_gb.groups.get(k, []))
        w_fo_xy.append((i, round(100 * k_w_fo_cnt / w_fo_cnt, 2), k_w_fo_cnt, k))
        wo_fo_xy.append((i, round(100 * k_wo_fo_cnt / wo_fo_cnt, 2), k_wo_fo_cnt, k))
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.4
    w_bars = ax.bar([i[0] for i in w_fo_xy], [i[1] for i in w_fo_xy], width=width, color='b', edgecolor='black', label='With Flash Order')
    wo_bars = ax.bar([i[0] + width for i in wo_fo_xy], [i[1] for i in wo_fo_xy], width=width, color='g', edgecolor='black', label='Without Flash Order')
    ax.bar_label(w_bars, labels=[f'{i[2]}\n({i[1]}%)' for i in w_fo_xy], fontsize=10)
    ax.bar_label(wo_bars, labels=[f'{i[2]}\n({i[1]}%)' for i in wo_fo_xy], fontsize=10)
    ax.set_xticks([i[0] + width / 2 for i in w_fo_xy])
    ax.set_xticklabels([i[3] for i in w_fo_xy], {'fontsize': 12})
    ax.set_title(f'Cases Distribution Breakdown', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    return


if __name__ == '__main__':
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    figure_7()
    figure_8()
    figure_9()
