import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from get_data import *
import tqdm
import glob
import datetime


class FlashOrderCalculator:
    def __init__(self, stock_code: str):
        """ Get a quotes df from and then compute fleeting quotes attributions:

            - DONE - Flash order spectrum (Freq plot by time)
            - DONE - Size of flash orders
            - HALF DONE - Cluster of flash orders
            - NOT YET - Bid/Ask flash orders discrepancy
            - DONE - Intra-day pattern of flash orders
            - DONE - Success rate of flash orders (Should be higher than Ashare/US market)

        :param stock_code: target stock to work on
        """
        self.data_gb = get_one_stock_data(stock_code=stock_code, verbose=True, gb_days=True)
        self.fod = {}  # flash order dict

    def classify(self, max_dur_ms: int = 100, trick_trade_thres_ms: int = 100):
        """ classify quote updates over the day, to look for flash orders

        :param max_dur_ms: Maximum duration allowed for the flash quotes, ms
        :param trick_trade_thres_ms: Maximum time in ms to count a trade that is tricked by the flash order
        :return:
        """
        key = f'[{max_dur_ms}|{trick_trade_thres_ms}]'
        if key in self.fod.keys():
            log_info(f'Classification of {key} done already')
            return
        one_fod = []
        for d, df in tqdm.tqdm(self.data_gb.items(), desc='ClassifyingFlashOrder', ncols=200, total=len(self.data_gb)):

            # Find improved and worsened cases to be considered
            bid_improved_ind = df.loc[(df['BP1'].diff() > 0) & (df['BP1'] > 0) & (df['BP1'].shift() > 0)].index.tolist()
            bid_worsened_ind = df.loc[(df['BP1'].diff() < 0) & (df['BP1'] > 0) & (df['BP1'].shift() > 0)].index.tolist()
            ask_improved_ind = df.loc[(df['SP1'].diff() < 0) & (df['SP1'] > 0) & (df['SP1'].shift() > 0)].index.tolist()
            ask_worsened_ind = df.loc[(df['SP1'].diff() > 0) & (df['SP1'] > 0) & (df['SP1'].shift() > 0)].index.tolist()
            bid_improved_ind_end = np.searchsorted(df['dt_ms'], df.loc[bid_improved_ind, 'dt_ms'] + max_dur_ms) - 1
            ask_improved_ind_end = np.searchsorted(df['dt_ms'], df.loc[ask_improved_ind, 'dt_ms'] + max_dur_ms) - 1
            bid_search_sec = zip(bid_improved_ind, bid_improved_ind_end, ['bid'] * len(bid_improved_ind))
            ask_search_sec = zip(ask_improved_ind, ask_improved_ind_end, ['ask'] * len(ask_improved_ind))

            one_day_fod = []
            for i_start, i_end, i_case in list(bid_search_sec) + list(ask_search_sec):

                # No updates within duration
                if i_start == i_end:
                    continue

                to_check_worsened = bid_worsened_ind if i_case == 'bid' else ask_worsened_ind
                p1_col = 'BP1' if i_case == 'bid' else 'SP1'
                v1_col = 'BV1' if i_case == 'bid' else 'SV1'

                # No worsening within duration after improvement
                tgt_ls = [i for i in to_check_worsened if i_start < i <= i_end]
                if not len(tgt_ls):
                    continue
                fo_end_ind = tgt_ls[0]

                # Trade hit on flash order
                fo_p1 = df.at[i_start, p1_col]
                if sum(df.loc[i_start:fo_end_ind, 'lastPx'] == fo_p1) > 0:
                    continue

                # price didn't go back to original level
                orig_price, fo_price = df.at[i_start - 1, p1_col], df.at[i_start, p1_col]
                if df.at[i_start - 1, p1_col] != df.at[fo_end_ind, p1_col]:
                    continue

                # Here assumed the qty at i_start is the flash order quantity
                qty = df.at[i_start, v1_col]

                one_day_fod.append({
                    'date': d,
                    'side': i_case,
                    'start': df.at[i_start, 'dt'],
                    'start_ms': df.at[i_start, 'dt_ms'],
                    'start_index': i_start,
                    'end': df.at[fo_end_ind, 'dt'],
                    'end_ms': df.at[fo_end_ind, 'dt_ms'],
                    'end_index': fo_end_ind,
                    'duration': df.at[fo_end_ind, 'dt_ms'] - df.at[i_start, 'dt_ms'],
                    'orig_price': orig_price,
                    'fo_price': fo_price,
                    'fo_qty': qty,
                    'tricked_trade': False,
                    'tricked': np.nan,
                    'tricked_index': np.nan,
                    'tricked_ms': np.nan,
                    'tricked_price': np.nan,
                    'tricked_qty': np.nan,
                    'tricked_ms_taken': np.nan,
                    'opp_tricked_trade': False,
                    'opp_tricked': np.nan,
                    'opp_tricked_index': np.nan,
                    'opp_tricked_ms': np.nan,
                    'opp_tricked_price': np.nan,
                    'opp_tricked_qty': np.nan,
                    'opp_tricked_ms_taken': np.nan
                })

            # Look for tricked trades
            check_end_ind = np.searchsorted(df['dt_ms'], [i['end_ms'] + trick_trade_thres_ms for i in one_day_fod]) - 1
            for ind, i_fo in zip(check_end_ind, one_day_fod):
                check_df = df.loc[i_fo['end_index']:ind]
                if check_df['lastPx'].isna().all():
                    continue
                check_col = 'BP1' if i_fo['side'] == 'bid' else 'SP1'
                check_sign = 1 if check_col == 'BP1' else -1
                check_price = ((check_df['lastPx'] - i_fo['orig_price']) * check_sign) <= 0
                check_opp_price = ((check_df['lastPx'] - i_fo['orig_price']) * check_sign) > 0
                if check_price.any():
                    tricked_ind = check_df.loc[check_price].index.tolist()[0]
                    i_fo.update({
                        'tricked_trade': True,
                        'tricked': check_df.loc[tricked_ind, 'dt'],
                        'tricked_index': tricked_ind,
                        'tricked_ms': check_df.loc[tricked_ind, 'dt_ms'],
                        'tricked_price': check_df.loc[tricked_ind, 'lastPx'],
                        'tricked_qty': check_df.loc[tricked_ind, 'size'],
                        'tricked_ms_taken': check_df.loc[tricked_ind, 'dt_ms'] - i_fo['end_ms']
                    })
                if check_opp_price.any():
                    opp_tricked_ind = check_df.loc[check_opp_price].index.tolist()[0]
                    i_fo.update({
                        'opp_tricked_trade': True,
                        'opp_tricked': check_df.loc[opp_tricked_ind, 'dt'],
                        'opp_tricked_index': opp_tricked_ind,
                        'opp_tricked_ms': check_df.loc[opp_tricked_ind, 'dt_ms'],
                        'opp_tricked_price': check_df.loc[opp_tricked_ind, 'lastPx'],
                        'opp_tricked_qty': check_df.loc[opp_tricked_ind, 'size'],
                        'opp_tricked_ms_taken': check_df.loc[opp_tricked_ind, 'dt_ms'] - i_fo['end_ms']
                    })
            one_fod += one_day_fod
        for i, i_fo in enumerate(one_fod):
            i_fo.update({'case_index': i})
        self.fod[key] = one_fod
        return

    def classify_spectrum(self, dur_start, dur_step, dur_end):
        dur_ls = range(dur_start, dur_end + 1, dur_step)
        for dur in dur_ls:
            log_info(f'Classifying flash orders of duration < {dur} ms')
            self.classify(max_dur_ms=dur)
        return

    def get_by_day_fo(self, duration):
        return {[i for i in self.fod[duration] if i['date'] == d] for d in self.data_gb.groups.keys()}

    def get_by_side_fo(self, duration):
        return {[i for i in self.fod[duration] if i['side'] == s] for s in ['bid', 'ask']}

    def plot_quote(self, duration, case_index):
        case = self.fod[duration][case_index]
        df = self.data_gb[case['date']]
        end_ind = case['end_index'] if not case['tricked_trade'] else case['tricked_index']
        end_ind = max(end_ind, -1 if not case['opp_tricked_trade'] else case['opp_tricked_index'])
        slice_df = df.loc[case['start_index'] - 5:end_ind + 5, :].copy()
        slice_df.loc[:, 'BP1'] = slice_df['BP1'].replace(0, np.nan).fillna(method='ffill')
        slice_df.loc[:, 'SP1'] = slice_df['SP1'].replace(0, np.nan).fillna(method='ffill')
        # slice_df = slice_df.loc[(slice_df['BP1'] > 0) & (slice_df['SP1'] > 0), :]

        # Start plotting
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(slice_df['dt'], slice_df['BP1'], color='gray', drawstyle='steps-post')
        ax.plot(slice_df['dt'], slice_df['SP1'], color='gray', drawstyle='steps-post')
        ax.axvline(slice_df.at[case['start_index'], 'dt'])
        ax.scatter(slice_df['dt'], slice_df['lastPx'], color='blue', label='Market Trades', alpha=0.5, s=40, marker='D')
        if case['tricked_trade']:
            row = slice_df.loc[case['tricked_index']]
            ax.scatter([row['dt']], [row['lastPx']], color='red', label='Tricked Trade', alpha=0.8, s=80, marker='D')
        if case['opp_tricked_trade']:
            row = slice_df.loc[case['opp_tricked_index']]
            ax.scatter([row['dt']], [row['lastPx']], color='violet', label='Opposite Tricked Trade', alpha=0.8, s=80, marker='D')
        ax.legend(loc='upper right')
        ax.set_title(f'{case["date"]} {case["side"]} duration={case["duration"]}ms tricked={case["tricked_trade"]}', fontsize=14)
        fig.tight_layout()
        plt.show()
        return
