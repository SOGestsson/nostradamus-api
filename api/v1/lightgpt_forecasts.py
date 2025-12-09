# inventory_algorithm/lightgpt_forecasts.py

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    from nixtla import NixtlaClient
    _HAS_LIGHTGPT = True
except Exception:
    _HAS_LIGHTGPT = False


class LightGPTForecast:
    """
    Batch forecaster for LightGPT with support for:
    - Multiple items in a single request
    - External drivers (regressors) like price, promotion, seasonality
    - Item-level attributes (brand, item_group, category, etc.)
    - Cross-item learning for related products
    
    Conventions:
    - Input history is a DataFrame with columns: ['item_id', 'day', 'actual_sale', ...drivers...]
    - Item attributes DataFrame with columns: ['item_id', 'brand', 'item_group', 'category', ...]
    - Drivers DataFrame with columns: ['day', 'driver_name', 'driver_value']
    """

    def __init__(self,
                 api_key: str | None = None,
                 model: str = 'lightgpt',  # e.g., 'lightgpt', 'lightgpt-advanced'
                 freq: str = 'D',  # 'D'=daily, 'MS'=monthly, 'W'=weekly
                 ):
        """
        Initialize LightGPT forecaster.
        
        Args:
            api_key: Nixtla API key (or set NIXTLA_API_KEY env var)
            model: LightGPT model variant
            freq: Data frequency ('D', 'MS', 'W', etc.)
        """
        self.model = model
        self.freq = freq
        self._client = None
        
        if not _HAS_LIGHTGPT:
            raise RuntimeError("Nixtla package not available. Install with `pip install nixtla`.")
        
        self._client = NixtlaClient(api_key=api_key or os.environ.get("NIXTLA_API_KEY"))

    # ---------- Batch forecast with drivers ----------
    def batch_forecast_with_drivers(self,
                                    hist: pd.DataFrame,
                                    item_attributes: pd.DataFrame | None = None,
                                    drivers: pd.DataFrame | None = None,
                                    forecast_periods: int = 30,
                                    exogenous_columns: List[str] | None = None) -> pd.DataFrame:
        """
        Generate batch forecasts for multiple items with external drivers and item attributes.
        
        Args:
            hist: Historical sales data
                Columns: ['item_id', 'day', 'actual_sale', ...optional driver columns...]
                Example: item_id, day, actual_sale, price, promotion, store_id
                
            item_attributes: Item-level metadata
                Columns: ['item_id', 'brand', 'item_group', 'category', ...]
                Used for cross-item learning and segmentation
                Example: item_id, brand, item_group, category, supplier, margin
                
            drivers: External drivers (optional)
                Columns: ['item_id', 'day', 'driver_name', 'driver_value']
                Or: ['day', 'driver_name', 'driver_value'] for global drivers
                Examples: price changes, promotions, holidays, stock levels
                
            forecast_periods: Number of periods to forecast
            
            exogenous_columns: List of driver/exogenous column names to use
                Example: ['price', 'promotion', 'seasonality_index']
                If None, auto-detects from hist columns
                
        Returns:
            DataFrame with columns:
                ['item_id', 'day', 'forecast', 'forecast_date']
        """
        try:
            print(f"Starting batch LightGPT forecast for {hist['item_id'].nunique()} items")
            
            # Prepare data
            df_hist = hist.copy()
            df_hist['day'] = pd.to_datetime(df_hist['day'])
            
            # Add item attributes if provided
            if item_attributes is not None:
                df_hist = df_hist.merge(item_attributes, on='item_id', how='left')
                print(f"  Added {len(item_attributes.columns)-1} item attributes")
            
            # Add drivers if provided
            if drivers is not None:
                df_drivers = drivers.copy()
                df_drivers['day'] = pd.to_datetime(df_drivers['day'])
                
                # If drivers have item_id, merge by item_id and day
                if 'item_id' in df_drivers.columns:
                    df_hist = df_hist.merge(df_drivers, on=['item_id', 'day'], how='left')
                else:
                    # Global drivers, merge only on day
                    df_hist = df_hist.merge(df_drivers, on='day', how='left')
                print(f"  Added external drivers")
            
            # Auto-detect exogenous columns if not provided
            if exogenous_columns is None:
                excluded = {'item_id', 'day', 'actual_sale', 'brand', 'item_group', 
                           'category', 'supplier', 'margin', 'sku', 'description'}
                exogenous_columns = [col for col in df_hist.columns if col not in excluded]
            
            # Prepare data for LightGPT
            # Rename columns to match Nixtla format
            df_formatted = df_hist[['item_id', 'day', 'actual_sale'] + exogenous_columns].copy()
            df_formatted.columns = ['unique_id', 'ds', 'y'] + exogenous_columns
            
            print(f"  Using exogenous columns: {exogenous_columns}")
            print(f"  Data shape: {df_formatted.shape}")
            
            # Call LightGPT
            fcst = self._client.forecast(
                df=df_formatted,
                h=forecast_periods,
                freq=self.freq,
                time_col='ds',
                target_col='y',
                model=self.model,
                X_df=df_formatted[['unique_id', 'ds'] + exogenous_columns] if exogenous_columns else None
            )
            
            # Format output
            result = pd.DataFrame({
                'item_id': fcst['unique_id'].astype(int),
                'day': pd.to_datetime(fcst['ds']),
                'forecast': fcst.get(self.model, fcst.get(f'{self.model}-q-50', np.nan)),
            })
            
            print(f"Batch forecast completed for {result['item_id'].nunique()} items")
            return result
            
        except Exception as e:
            print(f"Error in batch forecast: {str(e)}")
            raise

    # ---------- Multi-item forecast with cross-learning ----------
    def forecast_with_cross_learning(self,
                                     hist: pd.DataFrame,
                                     item_attributes: pd.DataFrame,
                                     group_column: str = 'brand',
                                     forecast_periods: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts with cross-learning by item groups (brand, category, etc.).
        Items in the same group share information for better forecasts.
        
        Args:
            hist: Historical sales data with columns ['item_id', 'day', 'actual_sale']
            item_attributes: Item metadata with columns ['item_id', group_column, ...]
            group_column: Column name for grouping (e.g., 'brand', 'category', 'supplier')
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with keys for each group containing grouped forecast results
        """
        try:
            print(f"Starting cross-learning forecast grouped by '{group_column}'")
            
            # Merge attributes with history
            df = hist.merge(item_attributes, on='item_id', how='left')
            
            results = {}
            groups = df[group_column].unique()
            
            for group in groups:
                print(f"  Forecasting group: {group}")
                
                # Filter data for this group
                group_data = df[df[group_column] == group].copy()
                
                # Prepare for batch forecast
                group_data_formatted = group_data[['item_id', 'day', 'actual_sale']].copy()
                group_data_formatted['day'] = pd.to_datetime(group_data_formatted['day'])
                group_data_formatted.columns = ['unique_id', 'ds', 'y']
                
                # Forecast this group
                fcst = self._client.forecast(
                    df=group_data_formatted,
                    h=forecast_periods,
                    freq=self.freq,
                    time_col='ds',
                    target_col='y',
                    model=self.model
                )
                
                results[group] = pd.DataFrame({
                    'item_id': fcst['unique_id'].astype(int),
                    'day': pd.to_datetime(fcst['ds']),
                    'forecast': fcst.get(self.model, np.nan),
                    'group': group
                })
            
            print(f"Cross-learning forecast completed for {len(groups)} groups")
            return results
            
        except Exception as e:
            print(f"Error in cross-learning forecast: {str(e)}")
            raise

    # ---------- Item-level hierarchical forecast ----------
    def hierarchical_forecast(self,
                             hist: pd.DataFrame,
                             item_attributes: pd.DataFrame,
                             hierarchy: List[str],  # e.g., ['brand', 'category', 'item_id']
                             forecast_periods: int = 30) -> pd.DataFrame:
        """
        Generate hierarchical forecasts respecting category structure.
        Ensures forecasts are coherent across hierarchy levels.
        
        Args:
            hist: Historical sales data
            item_attributes: Item hierarchy metadata
            hierarchy: List of columns defining hierarchy from top to bottom
                Example: ['brand', 'category', 'item_id']
            forecast_periods: Number of periods to forecast
            
        Returns:
            DataFrame with hierarchical forecasts
        """
        try:
            print(f"Starting hierarchical forecast with hierarchy: {' > '.join(hierarchy)}")
            
            df = hist.merge(item_attributes, on='item_id', how='left')
            
            # Create hierarchical identifier
            df['hierarchy_id'] = df[hierarchy].astype(str).agg('/'.join, axis=1)
            
            # Prepare for batch forecast
            df_formatted = df[['hierarchy_id', 'day', 'actual_sale']].copy()
            df_formatted['day'] = pd.to_datetime(df_formatted['day'])
            df_formatted.columns = ['unique_id', 'ds', 'y']
            
            # Forecast hierarchical structure
            fcst = self._client.forecast(
                df=df_formatted,
                h=forecast_periods,
                freq=self.freq,
                time_col='ds',
                target_col='y',
                model=self.model
            )
            
            # Parse hierarchy back
            result = pd.DataFrame({
                'hierarchy_id': fcst['unique_id'],
                'day': pd.to_datetime(fcst['ds']),
                'forecast': fcst.get(self.model, np.nan),
            })
            
            # Split hierarchy back into columns
            for i, level in enumerate(hierarchy):
                result[level] = result['hierarchy_id'].str.split('/').str[i]
            
            print(f"Hierarchical forecast completed")
            return result
            
        except Exception as e:
            print(f"Error in hierarchical forecast: {str(e)}")
            raise

    # ---------- Scenario analysis with drivers ----------
    def forecast_scenarios(self,
                          hist: pd.DataFrame,
                          scenarios: Dict[str, pd.DataFrame],
                          forecast_periods: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts under different scenarios (e.g., price changes, promotions).
        
        Args:
            hist: Historical sales data
            scenarios: Dictionary of scenario name -> driver values
                Example: {
                    'base_case': df_base_drivers,
                    'promotion': df_promo_drivers,
                    'price_increase': df_price_drivers
                }
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results for each scenario
        """
        try:
            print(f"Starting scenario analysis with {len(scenarios)} scenarios")
            
            results = {}
            
            for scenario_name, scenario_drivers in scenarios.items():
                print(f"  Forecasting scenario: {scenario_name}")
                
                # Combine historical data with scenario drivers
                scenario_data = hist.copy()
                scenario_data = scenario_data.merge(scenario_drivers, on=['item_id', 'day'], how='left')
                
                # Generate forecast for this scenario
                fcst = self.batch_forecast_with_drivers(
                    hist=scenario_data,
                    forecast_periods=forecast_periods
                )
                
                fcst['scenario'] = scenario_name
                results[scenario_name] = fcst
            
            print(f"Scenario analysis completed")
            return results
            
        except Exception as e:
            print(f"Error in scenario analysis: {str(e)}")
            raise

    # ---------- Item similarity and grouping ----------
    def get_similar_items(self,
                         item_attributes: pd.DataFrame,
                         reference_item_id: int,
                         similarity_columns: List[str],
                         top_n: int = 5) -> pd.DataFrame:
        """
        Find similar items based on attributes for cross-learning.
        
        Args:
            item_attributes: Item metadata
            reference_item_id: Item to find similar items for
            similarity_columns: Columns to use for similarity calculation
                Example: ['brand', 'category', 'price_range']
            top_n: Number of similar items to return
            
        Returns:
            DataFrame of similar items sorted by similarity score
        """
        try:
            ref_item = item_attributes[item_attributes['item_id'] == reference_item_id]
            
            if ref_item.empty:
                raise ValueError(f"Item {reference_item_id} not found")
            
            # Calculate similarity (matching attributes)
            similarity = []
            for _, item in item_attributes.iterrows():
                if item['item_id'] == reference_item_id:
                    continue
                
                # Count matching attributes
                matches = sum(1 for col in similarity_columns 
                            if item[col] == ref_item[col].iloc[0])
                
                similarity.append({
                    'item_id': item['item_id'],
                    'similarity_score': matches / len(similarity_columns)
                })
            
            result = pd.DataFrame(similarity).sort_values('similarity_score', ascending=False)
            return result.head(top_n)
            
        except Exception as e:
            print(f"Error finding similar items: {str(e)}")
            raise