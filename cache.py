import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd
from azure.data.tables import TableServiceClient, TableClient
from azure.core.exceptions import ResourceNotFoundError, AzureError

def serialize_datetime(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

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

class TableStorageCache:
    def __init__(self):
        self.connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.table_name = "PredictionCache"
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensure the cache table exists, create if it doesn't"""
        if not self.connection_string:
            logging.warning("No Azure Storage connection string provided. Caching disabled.")
            return

        try:
            table_service = TableServiceClient.from_connection_string(self.connection_string)
            table_service.create_table_if_not_exists(self.table_name)
        except AzureError as e:
            logging.error(f"Failed to initialize cache table: {str(e)}")

    def get_cache_key(self, date: str, country: str, interval_hours: int, use_llm: bool = False) -> str:
        """Generate a cache key"""
        llm_suffix = "_llm" if use_llm else ""
        return f"{country}_{date}_{interval_hours}{llm_suffix}"

    def get_cached_result(self, cache_key: str) -> dict:
        """Get cached result if exists and not expired"""
        if not self.connection_string:
            return None

        try:
            table_service = TableServiceClient.from_connection_string(self.connection_string)
            table_client = table_service.get_table_client(self.table_name)
            
            partition_key = cache_key[:50]  # Azure Table storage limit
            row_key = "1"  # Simple row key as we only need one record per prediction
            
            try:
                entity = table_client.get_entity(partition_key, row_key)
                cached_data = json.loads(entity['data'])
                expiry_time = datetime.fromisoformat(entity['expiry_time'])
                
                if expiry_time > datetime.utcnow():
                    return cached_data
                return None
            except ResourceNotFoundError:
                return None
                
        except AzureError as e:
            logging.error(f"Cache read error: {str(e)}")
            return None

    def set_cached_result(self, cache_key: str, data: dict, ttl_hours: int = 24):
        """Store result in cache with expiry time"""
        if not self.connection_string:
            return

        try:
            table_service = TableServiceClient.from_connection_string(self.connection_string)
            table_client = table_service.get_table_client(self.table_name)
            
            partition_key = cache_key[:50]  # Azure Table storage limit
            row_key = "1"
            expiry_time = datetime.utcnow() + timedelta(hours=ttl_hours)
            
            entity = {
                'PartitionKey': partition_key,
                'RowKey': row_key,
                'data': json.dumps(data),
                'expiry_time': expiry_time.isoformat()
            }
            
            table_client.upsert_entity(entity)
            
        except AzureError as e:
            logging.error(f"Cache write error: {str(e)}") 

    def get_historical_context(self, country: str, days: int = 30) -> List[Dict]:
        """
        Retrieve historical context for LLM analysis
        
        Args:
            country: Country to get historical data for
            days: Number of days of historical data to retrieve
        """
        try:
            if not self.connection_string:
                return []
            
            table_service = TableServiceClient.from_connection_string(self.connection_string)
            table_client = table_service.get_table_client(self.table_name)
            
            # Get recent predictions for the country
            query_filter = f"PartitionKey ge '{country}_' and PartitionKey lt '{country}`'"
            entities = table_client.query_entities(query_filter)
            
            historical_data = []
            for entity in entities:
                try:
                    data = json.loads(entity['data'])
                    if isinstance(data, dict) and 'Daily Forecasts' in data:
                        historical_data.append({
                            'date': entity.get('PartitionKey').split('_')[1],
                            'forecasts': data['Daily Forecasts'],
                            'actual': data.get('actual_orders', None)
                        })
                except Exception as e:
                    logging.warning(f"Failed to process historical entry: {str(e)}")
                    continue
                
            # Sort by date and limit to requested days
            historical_data.sort(key=lambda x: x['date'], reverse=True)
            return historical_data[:days]
            
        except Exception as e:
            logging.error(f"Failed to retrieve historical context: {str(e)}")
            return []