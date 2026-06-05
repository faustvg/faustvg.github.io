"""
generate_multi_series.py
========================
It pre-computes results for multiple time series indices and writes:

    assets/notebook_data_multi.js

Then commit that file to your repo — the portfolio page will automatically
gain a series-selector dropdown.

Usage in Colab:
    !python generate_multi_series.py

Estimated runtime: ~10–30 min depending on NN training speed.
"""
import sys, io, base64, json, os, warnings
warnings.filterwarnings('ignore')


# Use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Which series to pre-compute ───────────────────────────────────────────────
# Add or remove n values here (valid range: 1–414).
# More values = larger JS file and longer generation time.
N_VALUES = [5, 25, 50, 100, 150, 200, 250, 300, 350, 400]
H = 48          # forecast horizon (only h=48 is pre-computed)
DPI = 130       # PNG resolution — 130 is a good balance of quality vs file size

# ── Intercept matplotlib shows ────────────────────────────────────────────────
_mpl_captures = []

def _capture_plt_show(*args, **kwargs):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=DPI, bbox_inches='tight',
                facecolor='#111111', edgecolor='none')
    buf.seek(0)
    _mpl_captures.append('data:image/png;base64,' + base64.b64encode(buf.read()).decode())
    plt.close('all')

plt.show = _capture_plt_show

# ── Intercept Plotly shows ────────────────────────────────────────────────────
_plotly_captures = []

import plotly.graph_objects as _go

def _capture_go_show(self, *args, **kwargs):
    _plotly_captures.append(json.loads(self.to_json()))

_go.Figure.show = _capture_go_show

# Also cover IPython.display.display() used by Plotly in Colab/Jupyter
try:
    import IPython.display as _ipd
    _orig_display = _ipd.display

    def _capture_display(obj, *args, **kwargs):
        if isinstance(obj, _go.Figure):
            _plotly_captures.append(json.loads(obj.to_json()))
        else:
            _orig_display(obj, *args, **kwargs)

    _ipd.display = _capture_display
except Exception:
    pass

# ── Import project modules ────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from error_metrics_weights import *
from nn_2_models import *
from heuristic_models import *
from statisical_models import *
from functions import *

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading CSV data...')
df_train = pd.read_csv('Hourly_wdates.csv', index_col=0)
df_test  = pd.read_csv('Hourly-test.csv',   index_col=0)
print(f'Train: {df_train.shape}, Test: {df_test.shape}')
print(f'Computing {len(N_VALUES)} series: {N_VALUES}\n')

all_series = {}

for n in N_VALUES:
    sep = '=' * 55
    print(f'\n{sep}\n  Series n={n}, h={H}\n{sep}')

    train, test = prepare_data(n, df_train, df_test)

    # ── 1. Original time series ────────────────────────────────────────────
    _mpl_captures.clear()
    plot_original_data(train, test, n=n, h=H)
    png_original = _mpl_captures[0] if _mpl_captures else ''

    # ── 2. Heuristic models ────────────────────────────────────────────────
    print('  [1/4] Heuristic models (Holt-Winters, Seasonal Avg, Naive)...')
    hw_fc    = holt_winters(train, test, alpha=0.1, beta=0.1, gamma=0.1,
                             seasonal_periods=24, horizons=H)
    avg_fc   = seasonal_averaging_forecast(train, test[:H])
    naive_fc = seasonal_naive_forecast(train, test[:H], horizons=H)

    _mpl_captures.clear()
    plot_forecast(train, test,
                  hw_fc,    'Holt Winters',
                  avg_fc,   'Seasonal Averaging',
                  naive_fc, 'Naive Forecast',
                  n, H, 'Heuristic')
    png_heuristic  = _mpl_captures[0] if len(_mpl_captures) > 0 else ''
    png_heuristic2 = _mpl_captures[1] if len(_mpl_captures) > 1 else ''

    # ── 3. Statistical models (needed for all-models combination) ──────────
    print('  [2/4] Statistical models (ARIMA, BATS, SARIMAX)...')
    arima_fc   = arima(train, test, horizons=H)
    bats_fc    = bats(train, test, horizons=H)
    sarimax_fc = sarimax(train, test, horizons=H)
    # (no separate PNG tab for statistical models in the viewer)

    # ── 4. Neural network models ───────────────────────────────────────────
    print('  [3/4] Neural network models (N-BEATS, GRU, LSTM)...')
    nbeats_fc = nbeats_model(train, test, horizons=H)
    gru_fc    = gru_model(train, test, horizons=H)
    lstm_fc   = lstm_model(train, test, horizons=H)

    _mpl_captures.clear()
    plot_forecast(train, test,
                  nbeats_fc, 'N-Beats',
                  gru_fc,    'GRU',
                  lstm_fc,   'LSTM',
                  n, H, 'Neural Network')
    png_nn  = _mpl_captures[0] if len(_mpl_captures) > 0 else ''
    png_nn2 = _mpl_captures[1] if len(_mpl_captures) > 1 else ''

    # ── 5. All 9 models combined ───────────────────────────────────────────
    print('  [4/4] Evaluating all-model & best-3 combinations...')
    forecasts_all = {
        'N-BEATS':           nbeats_fc,
        'GRU':               gru_fc,
        'LSTM':              lstm_fc,
        'ARIMA':             arima_fc,
        'BATS':              bats_fc,
        'SARIMAX':           sarimax_fc,
        'Holt-Winters':      hw_fc,
        'Seasonal Average':  avg_fc,
        'Naive':             avg_fc,
    }

    _plotly_captures.clear()
    old_out = sys.stdout; sys.stdout = io.StringIO()
    evaluate_and_plot_forecasts_9(train, test[:H], forecasts_all, H, n, type='All Models')
    table_all = sys.stdout.getvalue(); sys.stdout = old_out

    plotly_all_fc  = _plotly_captures[0] if len(_plotly_captures) > 0 else {}
    plotly_all_cmp = _plotly_captures[1] if len(_plotly_captures) > 1 else {}

    # ── 6. Best 3 models (one per category) ───────────────────────────────
    f_heuristic   = {'Triple Exponential Smoothing': hw_fc,
                     'Seasonal Moving Average':       avg_fc,
                     'Seasonal Naive Method':         naive_fc}
    f_statistical = {'ARIMA': arima_fc, 'BATS': bats_fc, 'SARIMAX': sarimax_fc}
    f_nn          = {'N-Beats': nbeats_fc, 'GRU': gru_fc, 'LSTM': lstm_fc}

    _plotly_captures.clear()
    old_out = sys.stdout; sys.stdout = io.StringIO()
    evaluate_and_plot_forecasts_3(train, test[:H], H, f_heuristic, f_statistical, f_nn)
    table_best3 = sys.stdout.getvalue(); sys.stdout = old_out

    plotly_b3_fc  = _plotly_captures[0] if len(_plotly_captures) > 0 else {}
    plotly_b3_cmp = _plotly_captures[1] if len(_plotly_captures) > 1 else {}

    # ── Store ──────────────────────────────────────────────────────────────
    all_series[str(n)] = {
        'png_original':   png_original,
        'png_heuristic':  png_heuristic,   'png_heuristic2': png_heuristic2,
        'png_nn':         png_nn,           'png_nn2':        png_nn2,
        'table_all':      table_all,        'table_best3':    table_best3,
        'plotly_all_fc':  plotly_all_fc,    'plotly_all_cmp': plotly_all_cmp,
        'plotly_b3_fc':   plotly_b3_fc,     'plotly_b3_cmp':  plotly_b3_cmp,
    }

    n_png    = sum(1 for v in [png_original, png_heuristic, png_heuristic2, png_nn, png_nn2] if v)
    n_plotly = sum(1 for v in [plotly_all_fc, plotly_all_cmp, plotly_b3_fc, plotly_b3_cmp] if v)
    print(f'  ✓ Captured {n_png} PNGs, {n_plotly} Plotly charts')

# ── Write JS output ───────────────────────────────────────────────────────────
os.makedirs('assets', exist_ok=True)
out_path = os.path.join('assets', 'notebook_data_multi.js')

payload  = json.dumps({'available': N_VALUES, 'series': all_series}, separators=(',', ':'))
js_content = f'const NB_MULTI = {payload};\n'

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(js_content)

size_kb = os.path.getsize(out_path) // 1024
print(f'\n{"="*55}')
print(f'✓  Written:  {out_path}')
print(f'   Size:     {size_kb} KB')
print(f'   Series:   {len(all_series)} ({", ".join(str(k) for k in all_series)})')
print(f'\nNext step: commit assets/notebook_data_multi.js to your repo.')
print('The portfolio page will automatically show the series dropdown.')
