"""
Shared dependencies for API endpoints.

This module contains shared utilities and dependencies used across API endpoints.
"""
from typing import Dict
import pandas as pd
from api.models import SimInput


def build_dataframes(data: SimInput) -> Dict[str, pd.DataFrame]:
    """
    Build pandas DataFrames from input data with proper type coercion.
    
    Args:
        data: SimInput model containing historical sales and item configuration
        
    Returns:
        Dictionary of DataFrames for simulation input
    """
    # Build historical sales DataFrame
    df_his = pd.DataFrame(data.sim_input_his)

    # Ensure correct column types
    if 'day' in df_his.columns:
        # Convert dates to datetime format
        df_his['day'] = pd.to_datetime(df_his['day'].astype(str), errors='coerce')

    # Coerce item_id to integers (convert floats like 31.0 to 31)
    if 'item_id' in df_his.columns:
        item_series = pd.to_numeric(df_his['item_id'], errors='ignore')
        if pd.api.types.is_float_dtype(item_series):
            with pd.option_context('mode.use_inf_as_na', True):
                df_his['item_id'] = pd.to_numeric(df_his['item_id'], errors='coerce').astype('Int64')
        else:
            df_his['item_id'] = item_series
    
    if 'actual_sale' in df_his.columns:
        df_his['actual_sale'] = pd.to_numeric(df_his['actual_sale'], errors='coerce')

    # Select only required columns (preserve order)
    required_his_cols = ["item_id", "actual_sale", "day"]
    df_his = df_his[[col for col in required_his_cols if col in df_his.columns]]

    # Build items DataFrame
    df_items = pd.DataFrame(data.sim_rio_items)

    # Remove CSV artifacts (Unnamed columns and "0" column)
    if not df_items.empty:
        cols = [
            c for c in df_items.columns
            if not (str(c).startswith('Unnamed') or str(c) in ('Unnamed:', 'Unnamed: 0', '0'))
        ]
        df_items = df_items[cols]

    # Coerce numeric columns to proper types
    numeric_cols = ['min', 'max', 'actual_stock', 'ideal_stock', 'del_time', 'buy_freq', 'station']
    for col in numeric_cols:
        if col in df_items.columns:
            df_items[col] = pd.to_numeric(df_items[col], errors='coerce').fillna(0)

    # Ensure string columns are proper strings (keep station as numeric)
    string_cols = ['pn', 'description', 'purchasing_method']
    for col in string_cols:
        if col in df_items.columns:
            df_items[col] = df_items[col].astype(str)

    # Set column order (excluding Unnamed columns)
    wanted_items_cols = [
        col for col in [
            "pn", "description",
            "actual_stock", "ideal_stock", "station",
            "del_time", "buy_freq", "purchasing_method", "min", "max"
        ] if col in df_items.columns
    ]
    if wanted_items_cols:
        df_items = df_items[wanted_items_cols]

    # Build item details DataFrame
    df_item_details = pd.DataFrame(data.sim_rio_item_details)
    
    # Ensure string columns in item_details are strings
    if 'vendor_name' in df_item_details.columns:
        df_item_details['vendor_name'] = df_item_details['vendor_name'].astype(str)

    # Build on-order DataFrame
    raw_on_order = data.sim_rio_on_order
    # Handle empty or placeholder data → return empty DataFrame with standard columns
    if not raw_on_order:
        df_on_order = pd.DataFrame(columns=["pn", "est_deliv_date", "est_deliv_qty"])
    elif len(raw_on_order) > 0 and set(raw_on_order[0].keys()) == {"Empty", "DataFrame"}:
        df_on_order = pd.DataFrame(columns=["pn", "est_deliv_date", "est_deliv_qty"])
    else:
        df_on_order = pd.DataFrame(raw_on_order)
        # Fix dates and types in on_order
        if 'est_deliv_date' in df_on_order.columns and not df_on_order.empty:
            df_on_order['est_deliv_date'] = pd.to_datetime(df_on_order['est_deliv_date'].astype(str), errors='coerce')
        if 'est_deliv_qty' in df_on_order.columns:
            df_on_order['est_deliv_qty'] = pd.to_numeric(df_on_order['est_deliv_qty'], errors='coerce').fillna(0)
        if 'pn' in df_on_order.columns:
            df_on_order['pn'] = df_on_order['pn'].astype(str)

    return {
        "sim_input_his": df_his,
        "sim_rio_items": df_items,
        "sim_rio_item_details": df_item_details,
        "sim_rio_on_order": df_on_order,
    }
