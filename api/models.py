# api/models.py - Fixed version
"""
Pydantic models for API requests and responses.
"""
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, AliasChoices


class SimInput(BaseModel):
    """Input model for simulation data containing historical sales and item configuration."""
    sim_input_his: List[Dict[str, Any]]
    sim_rio_items: List[Dict[str, Any]]
    sim_rio_item_details: List[Dict[str, Any]]
    sim_rio_on_order: List[Dict[str, Any]]


class SimulationRequest(BaseModel):
    """Request model for simulation execution with configurable parameters."""
    sim_input_his: List[Dict[str, Any]]
    sim_rio_items: List[Dict[str, Any]]
    sim_rio_item_details: List[Dict[str, Any]]
    sim_rio_on_order: List[Dict[str, Any]]
    number_of_days: int = 900
    number_of_simulations: int = 1000
    service_level: float = 0.95


class ForecastRequest(BaseModel):
    """Request model for generating forecasts."""
    sim_input_his: List[Dict[str, Any]]  # Historical sales data with item_id, actual_sale, day
    forecast_periods: int = 30  # Number of periods to forecast
    mode: str = 'local'  # 'local' or 'timegpt'
    local_model: str = 'auto_arima'  # Model type for local mode
    season_length: int = 12  # Seasonality period
    freq: str = 'D'  # 'D'=daily, 'M'=monthly (month-start by convention), 'ME'=month-end, 'W'=weekly
    api_key: Optional[str] = None  # For TimeGPT mode
    quantiles: Optional[List[float]] = None  # For TimeGPT quantile forecasts

    # Optional tuning knobs for local_model='auto_model'
    auto_model_metric: Optional[str] = None  # 'robust'|'rmse'|'mae' (default handled in endpoint)
    auto_model_cv_h: Optional[int] = None  # CV horizon per window (periods)
    auto_model_n_windows: Optional[int] = None  # Number of rolling CV windows
    auto_model_lookback_days: Optional[int] = None  # Use only last N days per item
    auto_model_lookback_periods: Optional[int] = None  # Use only last N observations per item


class ForecastResponse(BaseModel):
    """Response model for forecast results."""
    item_id: int | str
    forecast: List[float]
    forecast_dates: List[str]
    model_used: str


class LightGPTForecastRequest(BaseModel):
    """Request model for LightGPT batch forecasting."""
    model_config = ConfigDict(populate_by_name=True)

    sim_input_his: List[Dict[str, Any]]  # Historical sales with item_id, day, actual_sale, optional drivers
    item_attributes: Optional[List[Dict[str, Any]]] = None  # Item metadata (brand, category, etc.)
    external_drivers: Optional[List[Dict[str, Any]]] = None  # External regressors (price, promotion, etc.)
    freq: str = 'D'  # 'D'=daily, 'M'=monthly (treated as month-start internally)
    forecast_periods: int = 30

    model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('model', 'nixtla_model', 'nixtlaModel'),
    )

    api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('api_key', 'apiKey', 'nixtla_api_key', 'nixtlaApiKey', 'NIXTLA_API_KEY'),
    )

    exogenous_columns: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices('exogenous_columns', 'exogenousColumns'),
    )

    forecast_type: str = 'batch'  # 'batch', 'cross_learning', 'hierarchical', 'scenarios'
    group_column: Optional[str] = None  # For cross_learning: 'brand', 'category', etc.
    hierarchy: Optional[List[str]] = None  # For hierarchical: ['brand', 'category', 'item_id']
    scenarios: Optional[Dict[str, List[Dict[str, Any]]]] = None  # For scenario analysis


class LightGPTResponse(BaseModel):
    """Response model for LightGPT forecasts."""
    forecasts: List[Dict[str, Any]]
    total_items: int
    forecast_type: str
    periods: int


class LightGBMTrainRequest(BaseModel):
        """Request model for training a monthly LightGBM model.

        Canonical format:
        - Wide history in `sim_input_his` with columns:
            item_id, day, actual_sale, plus optional wide regressors (support series).

        Optional:
        - `item_attributes` for static item metadata (category, brand, etc.).
        - `external_drivers` for long-format drivers; converted to wide on ingest.
        """

        customer_id: str
        store_root: Optional[str] = None

        sim_input_his: List[Dict[str, Any]]
        item_attributes: Optional[List[Dict[str, Any]]] = None
        external_drivers: Optional[List[Dict[str, Any]]] = None

        # Monthly-only for now; 'M' treated as month-start ('MS') internally
        freq: str = 'M'
        forecast_periods: int = 12

        # If provided, restrict which wide columns are treated as exogenous
        exogenous_columns: Optional[List[str]] = None

        # Registry
        model_version: Optional[str] = None
        status: str = 'staging'  # staging|prod|archived
        notes: Optional[str] = None

        # Guardrails
        min_history_points: int = 24
        min_improvement: float = 0.02


class LightGBMBatchForecastRequest(BaseModel):
        """Request model for batch forecasting with a trained LightGBM model."""

        customer_id: str
        store_root: Optional[str] = None

        sim_input_his: List[Dict[str, Any]]
        item_attributes: Optional[List[Dict[str, Any]]] = None
        external_drivers: Optional[List[Dict[str, Any]]] = None

        # Monthly-only for now; 'M' treated as month-start ('MS') internally
        freq: str = 'M'
        forecast_periods: int = 12
        exogenous_columns: Optional[List[str]] = None

        # Which registered model to use
        model_version: Optional[str] = None
        status: str = 'prod'


