from optimization import *
from flash_order import *
import warnings
warnings.filterwarnings('ignore')

# Serve as a cache for all the plots
path_df = get_path_df(update=False)

gb = path_df.groupby(['stock_code', 'foms', 'side', 'ts', 'tm'])

res_dict = calc_sim_results(path_df=path_df)


def figure_1():
    """ Stage 5: Heatmap of simulations of 0050, bid, tm=1-20, ts=1-20, with consideration of fo, foms=300 """
    save_path = os.path.join(PROJECT_DIR, 'report', 'fig', 'figure_1.png')
    one_dict = {k: v for k, v in res_dict.items() if k[0] == '0050' and k[1] == '300' and k[2] == 'bid'}
    plot_heatmap(stock_code_ls=['0050'], side_ls=['bid'], plot_attri_ls=['score_mean'], res_dict=one_dict, save_path=save_path)
    return


def figure_2():
    """ Stage 5: Cases distribution breakdown represented by a bar chart """
    save_path = os.path.join(PROJECT_DIR, 'report', 'fig', 'figure_2.png')
    stock_code, side, ts, tm, foms = '0050', 'bid', 20, 10, 300
    w_fo_path = os.path.join(PROJECT_DIR, f'optres/foms={foms}/stock_code={stock_code}/side={side}/ts={ts}/tm={tm}', 'full_res.pkl')
    wo_fo_path = w_fo_path.replace(f'foms={300}', 'foms=0')
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
    ax.set_xticks([i[0] + width / 2 for i in w_fo_xy], [i[3] for i in w_fo_xy], fontsize=14)
    ax.set_title(f'Cases Distribution Breakdown', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    return


if __name__ == '__main__':
    figure_1()
    figure_2()

