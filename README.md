# MAFS6100 Order Placement Strategy

---

`get_file.py`
- Contains function to call raw stock data

---

`helpers.py`
- Helper functions for faster processing

---

`simulation.py`
- Contains main function `sim_one_day_t2` to simulate an experiment on 1 day data
- `plot_one_sim` can be used visualize result of one trade 
- Can be executed from command line in:
```shell
python simulation.py --code=0050 --side=bid --ts=10 --tm=5 --overwrite
```

- Description:
```
usage: simulation.py [-h] [-C STOCK_CODE] [-S SIDE] [-s TS] [-m TM] [-o OVERWRITE]

optional arguments:
  -h, --help            show this help message and exit
  -C STOCK_CODE, --code STOCK_CODE Stock code to simulate
  -S SIDE, --side SIDE  bid or ask
  -s TS, --ts TS        T star in seconds
  -m TM, --tm TM        T mid in seconds
  -o OVERWRITE, --overwrite OVERWRITE Overwrite or not
```

---

`optimization.py`
- Call `simulation.py` to run simulation over different set of parameters
- Save results to `optres` such as `optres/stock_code=0050/ts=10/tm=5/yyyy-mm-dd.csv`
- Can be executed from command line in:

```shell
python optimization.py -C 0050 -S bid --ts_start=10 --ts_step=5 --ts_end=120 --tm_start=5 --tm_step=5 --tm_end=120
```

- Description:
```
usage: optimization.py [-h] [-C STOCK_CODE] [-S SIDE] [--ts_start TS_START] [--ts_step TS_STEP] [--ts_end TS_END] [--tm_start TM_START] [--tm_step TM_STEP] [--tm_end TM_END]

optional arguments:
  -h, --help            show this help message and exit
  -C STOCK_CODE, --code STOCK_CODE Stock code to simulate
  -S SIDE, --side SIDE  bid or ask
  --ts_start TS_START   T star start
  --ts_step TS_STEP     T star step
  --ts_end TS_END       T star end
  --tm_start TM_START   T mid start
  --tm_step TM_STEP     T mid step
  --tm_end TM_END       T mid end
```

- Or alternatively check out `stage-3.ipynb` to see how it can be called in notebook

---

### Notes

- `stage-2.ipynb` contains code to visualise score distribution of 1 simulation
- `stage3.ipynb` contains code to visualise heatmap of scores over different set of parameters
- Objective function is defined in `simulation.py`, and is assessed in `stage-3.ipynb`


