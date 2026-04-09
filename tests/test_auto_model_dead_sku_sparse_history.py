"""Regression: sparse monthly imports (no rows after last sale) must not forecast ETS/seasonal levels."""

import pandas as pd


def test_auto_model_monthly_sparse_history_long_dead_tail_forces_naive():
    """No explicit zero rows after last sale — regularization extends to current month."""
    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    # Six positive months ending mid-2020, then nothing in the file (typical FPro-style gaps).
    rows = [
        {'item_id': '101981_like', 'day': f'2020-{m:02d}-01', 'actual_sale': float(m)}
        for m in range(1, 7)
    ]
    df = pd.DataFrame(rows)

    forecaster = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = forecaster.auto_model_forecast_panel(df, h=6, metric='wape_bias', n_windows=2)

    assert str(out['model_used'].iloc[0]) == 'Naive'
    assert float(out['yhat'].max()) <= 0.0
