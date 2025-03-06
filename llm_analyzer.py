import os
from typing import Dict, List, Any
from openai import AsyncOpenAI
from datetime import datetime
import json
import logging

from dotenv import load_dotenv

load_dotenv()

class LLMAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def analyze_forecast(self, 
                             forecast_data: Dict[str, Any], 
                             historical_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze forecast data using LLM to provide insights and adjustments
        """
        try:
            # Prepare the prompt with forecast data and historical context
            prompt = self._create_analysis_prompt(forecast_data, historical_context)
            logging.info(f"Prompt: {prompt}")
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a sales forecasting analyst expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            # Parse the LLM response
            analysis = self._parse_llm_response(response.choices[0].message.content)
            
            # Combine original forecast with LLM insights
            enhanced_forecast = self._enhance_forecast(forecast_data, analysis)
            
            return enhanced_forecast
        
        except Exception as e:
            logging.error(f"LLM analysis failed: {str(e)}")
            return forecast_data

    def _create_analysis_prompt(self, 
                              forecast_data: Dict[str, Any], 
                              historical_context: List[Dict]) -> str:
        """Create a detailed prompt for the LLM analysis"""
        # Create CSV header and data for forecasts
        csv_data = "Date,Interval,Forecasted Orders,Lower CI,Upper CI\n"
        
        for date, intervals in forecast_data.get('Daily Forecasts', {}).items():
            for interval in intervals:
                csv_data += (
                    f"{date},{interval['Interval Start']},{interval['Forecasted Orders']},"
                    f"{interval['Confidence Interval']['Lower']},{interval['Confidence Interval']['Upper']}\n"
                )

        prompt = f"""
        Analyze the following sales forecast data and provide insights:

        Forecast Period: {forecast_data.get('Start Date')} to {forecast_data.get('End Date')}
        Country: {forecast_data.get('Country')}

        Forecast Data (CSV format):
        {csv_data}

        Provide your analysis in the following JSON format EXACTLY (no additional text before or after):
        {{
            "confidence_scores": {{
                // For each date in the forecast, provide a confidence score between 0 and 1
                // Example: "2024-12-25": 0.85
            }},
            "external_factors": [
                // List of strings describing external factors
                // Example: "Upcoming holiday season"
            ],
            "suggested_adjustments": {{
                // For each date, provide adjustment suggestions
                // Example: "2024-12-25": "Increase forecast by 10% due to holiday shopping"
            }},
            "risk_factors": [
                {{
                    "factor": "string describing the risk",
                    "impact": "High/Medium/Low",
                    "mitigation": "suggestion to mitigate the risk"
                }}
            ]
        }}

        Ensure your response is ONLY the JSON object, with no additional text.
        """
        return prompt

    def _format_historical_context(self, historical_context: List[Dict]) -> str:
        """Format historical data as CSV"""
        if not historical_context:
            return ""
            
        csv_data = "Historical Date,Interval,Forecasted Orders,Actual Orders\n"
        
        for entry in historical_context:
            date = entry['date']
            forecasts = entry.get('forecasts', {})
            actual = entry.get('actual')
            
            for interval_data in forecasts.values():
                for interval in interval_data:
                    csv_data += (
                        f"{date},{interval['Interval Start']},"
                        f"{interval['Forecasted Orders']},"
                        f"{actual if actual is not None else 'N/A'}\n"
                    )
        
        return f"\nHistorical Context (CSV format):\n{csv_data}"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format"""
        try:
            # Clean up the response
            # Remove any potential markdown code block markers
            response = response.replace('```json', '').replace('```', '')
            
            # Find the JSON content
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logging.error("No JSON object found in response")
                return {}
                
            json_str = response[json_start:json_end]
            
            # Remove any comments from the JSON string
            json_lines = json_str.split('\n')
            json_lines = [line for line in json_lines if not line.strip().startswith('//')]
            json_str = '\n'.join(json_lines)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}")
                # Try a more lenient parsing approach
                import ast
                # Convert single quotes to double quotes
                json_str = json_str.replace("'", '"')
                # Parse as Python literal and convert to JSON-compatible dict
                data = ast.literal_eval(json_str)
                return json.loads(json.dumps(data))
                
        except Exception as e:
            logging.error(f"Failed to parse LLM response: {str(e)}")
            logging.error(f"Raw response: {response}")
            return {}

    def _enhance_forecast(self, 
                         original_forecast: Dict[str, Any], 
                         llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine original forecast with LLM insights"""
        try:
            enhanced_forecast = original_forecast.copy()
            enhanced_forecast['llm_analysis'] = llm_analysis
            
            # Adjust confidence intervals based on LLM insights
            if 'Daily Forecasts' in enhanced_forecast and 'confidence_scores' in llm_analysis:
                for date, intervals in enhanced_forecast['Daily Forecasts'].items():
                    # Get confidence score, default to 1.0 if not found or invalid
                    try:
                        confidence_score = float(llm_analysis['confidence_scores'].get(date, 1.0))
                    except (TypeError, ValueError):
                        logging.warning(f"Invalid confidence score for date {date}, using default 1.0")
                        confidence_score = 1.0

                    for interval in intervals:
                        try:
                            # Ensure we have valid numeric values
                            lower_ci = float(interval['Confidence Interval']['Lower'])
                            upper_ci = float(interval['Confidence Interval']['Upper'])
                            
                            ci_width = upper_ci - lower_ci
                            
                            # Adjust CI based on LLM confidence score
                            adjustment = (ci_width * (1 - confidence_score) * 0.5)
                            
                            interval['Confidence Interval']['Lower'] = max(0, lower_ci - adjustment)
                            interval['Confidence Interval']['Upper'] = upper_ci + adjustment
                            
                        except (TypeError, ValueError, KeyError) as e:
                            logging.warning(f"Failed to adjust confidence interval: {str(e)}")
                            continue

            return enhanced_forecast
            
        except Exception as e:
            logging.error(f"Failed to enhance forecast: {str(e)}")
            # Return original forecast if enhancement fails
            return original_forecast