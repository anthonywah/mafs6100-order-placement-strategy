import numpy as np

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

    def classify(self, max_dur_ms: int):
        """ classify quote updates over the day, to look for flash orders

        :param max_dur_ms: Maximum duration allowed for the flash quotes
        :return:
        """
        one_fod = []
        for d, df in tqdm.tqdm(self.data_gb.items(), desc='ClassifyingFlashOrder', ncols=200, total=len(self.data_gb)):

            bid_improved_ind = df.loc[df['BP1'].diff() > 0].index.tolist()
            bid_worsened_ind = df.loc[df['BP1'].diff() < 0].index.tolist()
            ask_improved_ind = df.loc[df['SP1'].diff() < 0].index.tolist()
            ask_worsened_ind = df.loc[df['SP1'].diff() > 0].index.tolist()

            # Flash bid order
            bid_improved_ind_end = np.searchsorted(df['dt_ms'], df.loc[bid_improved_ind, 'dt_ms'] + max_dur_ms)
            bid_search_se = zip(bid_improved_ind, [i - 1 for i in bid_improved_ind_end])
            for i_start, i_end in bid_search_se:

                # No updates within duration
                if i_start == i_end:
                    continue

                # No worsening within duration after improvement
                tgt_ls = [i for i in bid_worsened_ind if i_start < i <= i_end]
                if not len(tgt_ls):
                    continue
                fo_end_ind = tgt_ls[0]

                # Trade hit on flash order
                fo_bid = df.at[i_start, 'BP1']
                if sum(df.loc[i_start:fo_end_ind, 'lastPx'] == fo_bid) > 0:
                    continue

                # price didn't go back to original level
                orig_price, fo_price = df.at[i_start - 1, 'BP1'], df.at[i_start, 'BP1']
                if df.at[i_start - 1, 'BP1'] != df.at[fo_end_ind, 'BP1']:
                    continue

                # Here assumed the qty at i_start is the flash order quantity
                qty = df.at[i_start, 'BV1']

                one_fod.append({
                    'date': d,
                    'side': 'bid',
                    'start': df.at[i_start, 'dt'],
                    'start_ms': df.at[i_start, 'dt_ms'],
                    'start_index': i_start,
                    'end': df.at[fo_end_ind, 'dt'],
                    'end_ms': df.loatac[fo_end_ind, 'dt_ms'],
                    'end_index': fo_end_ind,
                    'duration': df.at[i_start, 'dt_ms'] - df.at[fo_end_ind, 'dt_ms'],
                    'orig_price': orig_price,
                    'fo_price': fo_price,
                    'fo_qty': qty
                })