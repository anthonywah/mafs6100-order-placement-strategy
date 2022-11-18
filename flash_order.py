import numpy as np
import matplotlib.pyplot as plt
from get_data import *
import tqdm
import glob
import datetime


class FlashOrderCalculator:
    def __init__(self, stock_code: str):
        """ Get a quotes df from and then compute fleeting quotes attributions:

            - Flash order spectrum (Freq plot by time)
            - Size of flash orders
            - Cluster of flash orders
            - Bid/Ask flash orders discrepancy
            - Intra-day pattern of flash orders
            - Success rate of flash orders (Should be higher than Ashare/US market)

        :param stock_code: target stock to work on
        """
        self.data_gb = get_one_stock_data(stock_code=stock_code, verbose=True, gb_days=True)
        self.fod = {}  # flash order dict

    def classify(self, max_dur_ms: int = 100):
        """ classify quote updates over the day, to look for flash orders

        :param max_dur_ms: Maximum duration allowed for the flash quotes
        :return:
        """
        one_fod = []
        for d, df in tqdm.tqdm(self.data_gb.items(), desc='ClassifyingFlashOrder', ncols=200, total=len(self.data_gb)):

            bid_improved_ind = df.loc[(df['BP1'].diff() > 0) & (df['BP1'] > 0) & (df['BP1'].shift() > 0)].index.tolist()
            bid_worsened_ind = df.loc[(df['BP1'].diff() < 0) & (df['BP1'] > 0) & (df['BP1'].shift() > 0)].index.tolist()
            ask_improved_ind = df.loc[(df['SP1'].diff() < 0) & (df['SP1'] > 0) & (df['SP1'].shift() > 0)].index.tolist()
            ask_worsened_ind = df.loc[(df['SP1'].diff() > 0) & (df['SP1'] > 0) & (df['SP1'].shift() > 0)].index.tolist()

            # Flash bid order
            bid_improved_ind_end = np.searchsorted(df['dt_ms'], df.loc[bid_improved_ind, 'dt_ms'] + max_dur_ms)
            ask_improved_ind_end = np.searchsorted(df['dt_ms'], df.loc[ask_improved_ind, 'dt_ms'] + max_dur_ms)
            bid_search_sec = zip(bid_improved_ind, [i - 1 for i in bid_improved_ind_end], ['bid'] * len(bid_improved_ind))
            ask_search_sec = zip(ask_improved_ind, [i - 1 for i in ask_improved_ind_end], ['ask'] * len(ask_improved_ind))
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

                one_fod.append({
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
                    'fo_qty': qty
                })
        self.fod[max_dur_ms] = one_fod
        return


def plot_quote(quotes, index_time):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(quotes['dt'], quotes['BP1'], color='gray', drawstyle='steps-post')
    ax.plot(quotes['dt'], quotes['SP1'], color='gray', drawstyle='steps-post')
    ax.axvline(index_time, color='violet')
    ax.scatter(quotes['dt'], quotes['lastPx'], color='blue', label='Market Trades', alpha=0.7, s=80, marker='D')
    ax.legend(loc='upper right')
    plt.show()
    return


# for case_ind in range(a.fod[100].__len__()):
#     case = a.fod[100][case_ind]
#     if case['date'] > '2022-01-10':
#         break
#     df = a.data_gb[case['date']]
#
#     slice_df = df.loc[case['start_index'] - 5:case['end_index'] + 5, :]
#     slice_df = slice_df.loc[(slice_df['BP1'] > 0) & (slice_df['SP1'] > 0), :]
#     slice_ind_time = df.loc[case['start_index'] - 1, 'dt']
#     plot_quote(slice_df, index_time=slice_ind_time)
#     print(case['date'], case['start_index'])

