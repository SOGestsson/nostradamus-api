import pandas as pd

from inventory_algorithm.lightgpt_forecasts import LightGPTForecast, _canonicalize_freq, _nixtla_compat_freq


def test_nixtla_compat_freq_maps_ms_to_m():
    assert _nixtla_compat_freq('MS') == 'M'
    assert _nixtla_compat_freq('ms') == 'M'
    assert _nixtla_compat_freq('M') == 'M'
    assert _nixtla_compat_freq('D') == 'D'


def test_monthly_internal_freq_is_ms_but_label_is_m():
    f = LightGPTForecast(freq='M', api_key='test-key')
    assert f.freq == 'MS'
    assert f.freq_label == 'M'


def test_me_legacy_alias_maps_to_month_start():
    assert _canonicalize_freq('ME') == 'MS'
    f = LightGPTForecast(freq='ME', api_key='test-key')
    assert f.freq == 'MS'


def test_monthly_output_alignment_is_month_start():
    ds = pd.to_datetime(pd.Series(['2025-01-15', '2025-02-28']))
    aligned = ds.dt.to_period('M').dt.to_timestamp(how='start')
    assert aligned.iloc[0].day == 1
    assert aligned.iloc[1].day == 1
