import logging
import os
import json
import pickle
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, date

import azure.functions as func
import pandas as pd
from azure.data.tables import TableServiceClient, TableClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
import numpy as np
import aiohttp

from model import predict_future_orders, predict_total_orders
from data_loader import load_data
from cache import TableStorageCache
from llm_analyzer import LLMAnalyzer
import asyncio

app = func.FunctionApp()

# Initialize cache
cache = TableStorageCache()

def serialize_datetime(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def process_forecast_for_cache(forecast_data: Dict) -> Dict:
    """Process forecast data to ensure it's cache-friendly"""
    processed = forecast_data.copy()
    
    # Convert all Interval Start times to ISO format strings
    if 'Intervals' in processed:
        for interval in processed['Intervals']:
            if isinstance(interval.get('Interval Start'), (datetime, pd.Timestamp)):
                interval['Interval Start'] = interval['Interval Start'].isoformat()
    
    return processed

def process_cached_forecast(cached_data: Dict) -> Dict:
    """Process cached forecast data back to the correct format"""
    processed = cached_data.copy()
    
    # Convert ISO format strings back to datetime objects
    if 'Intervals' in processed:
        for interval in processed['Intervals']:
            if isinstance(interval.get('Interval Start'), str):
                interval['Interval Start'] = pd.to_datetime(interval['Interval Start'])
    
    return processed

def create_error_response(message: str, status_code: int = 400) -> func.HttpResponse:
    logging.error(f"Error: {message}")
    return func.HttpResponse(
        json.dumps({"error": message}),
        status_code=status_code,
        mimetype="application/json"
    )

def validate_request_data(req: func.HttpRequest) -> Dict[str, Any]:
    try:
        req_body = req.get_json()
        data = req_body.get('data')
        if not data:
            raise ValueError("Please provide data in the request body")
        return data
    except ValueError as e:
        raise ValueError(f"Invalid request data: {str(e)}")

@app.route(route="predict-next-order-city", auth_level=func.AuthLevel.FUNCTION)
def predict_next_order_city(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = validate_request_data(req)
        
        # Extract and validate parameters
        start_date_time = data.get('start_date_time')
        country = data.get('country')
        hours_ahead = data.get('hours_ahead', 2)
        
        if not all([start_date_time, country]):
            return create_error_response(
                "Missing required parameters: start_date_time and country"
            )
        
        predictions = predict_future_orders(start_date_time, country, hours_ahead)
        return func.HttpResponse(
            json.dumps(predictions, default=str),
            mimetype="application/json"
        )
        
    except Exception as e:
        return create_error_response(str(e))

async def fetch_current_orders(country: str) -> Dict[str, float]:
    """Fetch current orders from the webhook."""
    webhook_url = f"https://webhook-ai.bythjul.net/webhook/dcc1b4a4-0578-4223-bd06-1f952f58bf7b"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(webhook_url, params={"country": country}) as response:
                response.raise_for_status()
                data = await response.json()
                return {
                    "current": float(data["current"]),
                    "previous_hour": float(data["previous_hour"])
                }
    except Exception as e:
        logging.error(f"Failed to fetch current orders from webhook: {str(e)}")
        raise ValueError(f"Failed to fetch current orders: {str(e)}")

@app.route(route="predict-sales", auth_level=func.AuthLevel.FUNCTION)
async def predict_sales(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = validate_request_data(req)
        # set key to lower case
        data = {k.lower(): v for k, v in data.items()}
        
        # Extract and validate parameters
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        date = data.get('date')  # Keep for backward compatibility
        country = data.get('country')
        interval_hours = data.get('interval_hours', 3)
        skip_cache = data.get('skip_cache', False)
        
        # New parameters for real-time adjustments
        reference_timestamp = data.get('reference_timestamp')
        current_orders = data.get('current_orders')
        
        # If reference_timestamp and current_orders are not provided, fetch from webhook
        if not reference_timestamp and not current_orders:
            try:
                current_orders = await fetch_current_orders(country)
                reference_timestamp = datetime.now().isoformat()
                logging.info(f"Fetched current orders from webhook: {current_orders}")
            except Exception as e:
                logging.warning(f"Failed to fetch current orders from webhook: {str(e)}")
                # Continue without real-time data if fetch fails
                current_orders = None
                reference_timestamp = None
        
        # Validate inputs
        if not country:
            return create_error_response("Missing required parameter: country")
            
        # Handle both single date and date span scenarios
        if date:
            prediction_date = date
            prediction_end_date = date
        elif start_date and end_date:
            prediction_date = start_date
            prediction_end_date = end_date
        else:
            return create_error_response(
                "Must provide either 'date' or both 'start_date' and 'end_date'"
            )
        
        # Validate real-time parameters if provided
        if bool(reference_timestamp) != bool(current_orders):
            return create_error_response(
                "Both reference_timestamp and current_orders must be provided together"
            )
        
        if current_orders and not isinstance(current_orders, dict):
            return create_error_response(
                "current_orders must be a dictionary with 'current' and 'previous_hour' keys"
            )
        
        if current_orders and not all(k in current_orders for k in ['current', 'previous_hour']):
            return create_error_response(
                "current_orders must contain both 'current' and 'previous_hour' values"
            )
        
        # Validate numeric values in current_orders
        if current_orders:
            try:
                current_orders['current'] = float(current_orders['current'])
                current_orders['previous_hour'] = float(current_orders['previous_hour'])
            except (ValueError, TypeError):
                return create_error_response(
                    "current_orders values must be valid numbers"
                )
        
        # Check cache first if skip_cache is False and no real-time data
        if not skip_cache and not current_orders:
            cache_key = cache.get_cache_key(f"{prediction_date}-{prediction_end_date}", country, interval_hours)
            cached_result = cache.get_cached_result(cache_key)
            
            if cached_result:
                logging.info(f"Cache hit for key: {cache_key}")
                return func.HttpResponse(
                    json.dumps(cached_result, default=serialize_datetime),
                    mimetype="application/json"
                )
        
        try:
            # Generate new prediction
            logging.info(f"{'Skipping cache' if skip_cache else 'Cache miss'} for prediction")
            forecast_df, daily_summary = predict_total_orders(
                prediction_date, 
                country, 
                end_date=prediction_end_date,
                include_fifth_interval=True,
                interval_hours=interval_hours,
                reference_timestamp=reference_timestamp,
                current_orders=current_orders
            )
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return create_error_response(f"Failed to generate prediction: {str(e)}")
        
        try:
            # Group intervals by date
            forecast_df['Date'] = pd.to_datetime(forecast_df['Interval Start']).dt.date
            grouped_forecast = {}
            
            for date, group in forecast_df.groupby('Date'):
                formatted_intervals = [{
                    "Interval Start": row['Interval Start'].isoformat(),
                    "Forecasted Orders": float(row['Forecasted Orders']),
                    "Confidence Interval": {
                        "Lower": float(row['Lower CI']),
                        "Upper": float(row['Upper CI'])
                    }
                } for _, row in group.iterrows()]
                
                grouped_forecast[str(date)] = formatted_intervals
            
            response_data = {
                "Start Date": prediction_date,
                "End Date": prediction_end_date,
                "Country": country,
                "Daily Forecasts": grouped_forecast
            }
            
            # Add daily summary if available
            if daily_summary:
                response_data["Daily Summary"] = daily_summary
            
            # Enhance forecast with LLM analysis if requested
            use_llm = data.get('use_llm', False)
            if use_llm:
                llm_analyzer = LLMAnalyzer()
                historical_context = cache.get_historical_context(country)
                enhanced_forecast = await llm_analyzer.analyze_forecast(
                    response_data,
                    historical_context
                )
                response_data = enhanced_forecast
            
            # Only cache if no real-time data was used
            if not current_orders:
                cache_key = cache.get_cache_key(
                    f"{prediction_date}-{prediction_end_date}", 
                    country, 
                    interval_hours,
                    use_llm=use_llm
                )
                cache.set_cached_result(cache_key, response_data)
            
            return func.HttpResponse(
                json.dumps(response_data, default=serialize_datetime),
                mimetype="application/json"
            )
            
        except Exception as e:
            logging.error(f"Response formatting error: {str(e)}")
            return create_error_response(f"Failed to format response: {str(e)}")
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return create_error_response(str(e))