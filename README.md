# MAFS6100 Order Placement Strategy

### To set up this repo:

(Path to data is specified in `get_data.py` line 12. You may change it to your own local path)

Run `pip install -r requirements.txt` on your python / virtual environment to install all the dependencies first. Then follow the below steps to generate figures in the report.

- Firstly download contents of `cache` and `optres` directory [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cswahaa_connect_ust_hk/Emzxtl_hhlJDpjJDTOFIwW0BsvBY3MZoxfWWZ0m79ToWGA?e=BTg4RW):
- Unzip the content of these files and place them inside `cache` and `optres` respectively. 
- Then run cd to this project directory and run `python setup.py` to ensure necessary directories and files exist
- Lastly run `python run_figures.py` to generate all the figures in the report (report/final_report.pdf)

Details of usage of different scripts are listed below

---

`get_data.py`
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
  -F FOMS, --foms FOMS  flash order ms
  -s TS, --ts TS        T star in seconds
  -m TM, --tm TM        T mid in seconds
  -o OVERWRITE, --overwrite OVERWRITE Overwrite or not
```

---

`optimization.py`
- Call `simulation.py` to run simulation over different set of parameters
- Save results to `optres` such as `optres/foms=0/stock_code=0050/ts=10/tm=5/yyyy-mm-dd.csv`
- Can be executed from command line in:

```shell
python optimization.py -C 0050 -S bid -F 0 --ts_start=10 --ts_step=5 --ts_end=120 --tm_start=5 --tm_step=5 --tm_end=120
```

- Description:
```
usage: optimization.py [-h] [-C STOCK_CODE] [-S SIDE] [--ts_start TS_START] [--ts_step TS_STEP] [--ts_end TS_END] [--tm_start TM_START] [--tm_step TM_STEP] [--tm_end TM_END]

optional arguments:
  -h, --help            show this help message and exit
  -C STOCK_CODE, --code STOCK_CODE Stock code to simulate
  -S SIDE, --side SIDE  bid or ask
  -F FOMS, --foms FOMS  flash order ms
  -o OVERWRITE, --overwrite OVERWRITE Overwrite or not
  --ts_start TS_START   T star start
  --ts_step TS_STEP     T star step
  --ts_end TS_END       T star end
  --tm_start TM_START   T mid start
  --tm_step TM_STEP     T mid step
  --tm_end TM_END       T mid end
```

- Or alternatively check out `stage-3.ipynb` to see how it can be called in notebook

---
