from azure_connector import AzureConnector
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
import logging
import datetime
import warnings

logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("neuralprophet").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_holidays():
    """Get holidays dataframe matching the training format"""
    holidays_df = pd.DataFrame({
        "event": 'Holiday',
        "ds": pd.to_datetime([
            "2023-03-21", "2023-04-20", "2023-04-21", "2023-04-27", "2023-05-01",
            "2023-06-28", "2023-06-29", "2023-09-25", "2023-12-01", "2023-12-16",
            "2023-12-25", "2023-12-26", "2024-01-01", "2024-03-21", "2024-03-29",
            "2024-04-01", "2024-04-27", "2024-05-01", "2024-06-16", "2024-06-17",
            "2024-08-09", "2024-09-24", "2024-12-16", "2024-12-25", "2024-12-26",
            "2025-01-01", "2025-03-21", "2025-04-18", "2025-04-21", "2025-04-27",
            "2025-04-28", "2025-05-01", "2025-06-16", "2025-08-09", "2025-09-24",
            "2025-12-16", "2025-12-25", "2025-12-26"
        ]),
        "lower_window": 0,
        "upper_window": 0
    })
    return holidays_df

def prepare_forecast_data(historical_df, forecast_periods=7):
    """
    Prepare data for forecasting by creating a future dataframe
    that includes both historical context and future dates
    """
    # Ensure ds is datetime
    historical_df['ds'] = pd.to_datetime(historical_df['ds'])
    
    # Get the last date from historical data
    last_date = historical_df['ds'].max()
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_periods,
        freq='D'
    )
    
    # For NeuralProphet with lags, we need historical context
    # Include the last 30 days (n_lags) + future dates
    context_start = last_date - pd.Timedelta(days=29)  # 30 days including last_date
    
    # Create complete date range (context + future)
    complete_dates = pd.date_range(
        start=context_start,
        end=future_dates.max(),
        freq='D'
    )
    
    # Create the prediction dataframe
    forecast_df = pd.DataFrame({'ds': complete_dates})
    
    # Add historical values where they exist
    forecast_df = forecast_df.merge(
        historical_df[['ds', 'y']], 
        on='ds', 
        how='left'
    )
    
    print(f"Created forecast dataframe:")
    print(f"- Date range: {forecast_df['ds'].min()} to {forecast_df['ds'].max()}")
    print(f"- Total days: {len(forecast_df)}")
    print(f"- Historical days: {forecast_df['y'].notna().sum()}")
    print(f"- Future days: {forecast_df['y'].isna().sum()}")
    
    return forecast_df

def generate_forecast_old(model, historical_df, holidays_df, forecast_periods=7):
    """
    Generate forecast using the loaded MLflow model
    """
    try:
        # Prepare the forecast dataframe
        forecast_input = prepare_forecast_data(historical_df, forecast_periods)
        
        # Add holidays to the forecast period
        forecast_start = historical_df['ds'].max() + pd.Timedelta(days=1)
        forecast_end = forecast_start + pd.Timedelta(days=forecast_periods-1)
        
        relevant_holidays = holidays_df[
            (holidays_df['ds'] >= forecast_input['ds'].min()) &
            (holidays_df['ds'] <= forecast_input['ds'].max())
        ]
        
        print(f"Found {len(relevant_holidays)} holidays in forecast period")
        
        # If we have holidays, we need to add them to the input
        if len(relevant_holidays) > 0:
            # For MLflow model, we might need to handle holidays differently
            # Let's try with just the basic forecast_input first
            pass
        
        # Make prediction using MLflow model
        print("Making prediction...")
        forecast_result = model.predict(forecast_input[['ds']])
        
        # Filter to only future predictions
        future_forecast = forecast_result[
            forecast_result['ds'] >= forecast_start
        ].copy()
        
        # Add metadata
        future_forecast['Created Date'] = datetime.datetime.now()
        # future_forecast['Model_Version'] = "NeuralProphet_7_day_v3"
        # future_forecast['Forecast_Type'] = "7_Day_Ahead"
        
        return future_forecast
        
    except Exception as e:
        print(f"Error in generate_forecast: {e}")
        # Fallback: try with minimal input
        try:
            print("Trying fallback approach...")
            last_date = historical_df['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            simple_input = pd.DataFrame({'ds': future_dates})
            forecast_result = model.predict(simple_input)
            forecast_result['Created_Date'] = datetime.datetime.now()
            return forecast_result
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise e2

#THIS IS THE WORKING FUNCTION 8/7/2025
def generate_forecast(model, historical_df, forecast_periods=7):
    """
    Generate forecast using the corrected MLflow model wrapper
    """
    try:
        # Ensure proper data types
        historical_df['ds'] = pd.to_datetime(historical_df['ds'])
        
        # Use the model's built-in method for creating forecast input
        # This method is now available in our corrected wrapper
        forecast_input = model.unwrap_python_model().make_future_dataframe(
            historical_df=historical_df,
            periods=forecast_periods,
            freq='D'
        )
        
        print(f"Forecast input prepared:")
        print(f"- Total rows: {len(forecast_input)}")
        print(f"- Date range: {forecast_input['ds'].min()} to {forecast_input['ds'].max()}")
        print(f"- Historical rows with y: {forecast_input['y'].notna().sum()}")
        print(f"- Future rows to predict: {forecast_input['y'].isna().sum()}")
        
        # Make prediction
        print("Making prediction with corrected wrapper...")
        forecast_result = model.predict(forecast_input)
        
        # Filter to only future predictions
        last_historical_date = historical_df['ds'].max()
        future_forecast = forecast_result[
            forecast_result['ds'] >= last_historical_date
        ].copy()
        
        # Add metadata
        future_forecast['Created Date'] = datetime.datetime.now()
        # future_forecast['Model_Version'] = "NeuralProphet_7_day_Fixed"
        # future_forecast['Forecast_Type'] = "7_Day_Ahead"
        
        print(f"Forecast completed: {len(future_forecast)} future predictions")
        
        return future_forecast
    
    except Exception as e:
        print(f"Error in generate_forecast_corrected: {e}")
        import traceback
        traceback.print_exc()
        raise e

def flatten_forecast(df):
    rows = []
    # make sure ds is a datetime
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    
    for _, row in df.iterrows():
        base = row['ds']
        for h in range( len(df)):  
            point = row.get(f'yhat{h}')
            if pd.notna(point):
                rows.append({
                    #'ds': base + pd.Timedelta(days=h-1),
                    'ds':row.get('ds'),
                    'Forecast': point,
                    'yhat_5%': row.get(f'yhat{h} 5.0%'),
                    'yhat_95%': row.get(f'yhat{h} 95.0%'),
                    'ar': row.get(f'ar{h}'),
                    'trend': row.get(f'trend'),
                    'weekly_seasonality': row.get(f'season_weekly'),
                    'Holiday_effect': row.get(f'event_Holiday'),
                    'Created Date':datetime.datetime.now()
                })
    return pd.DataFrame(rows)

def recursive_predict_old(model, history_df, total_days=30, chunk_size=7, holidays_df=None):
    """
    Roll forward a NeuralProphet 7-day model to get `total_days` of forecasts
    in `chunk_size`-day increments.
    """
    # Work with a copy of history
    history = history_df[['ds','y']].sort_values('ds').reset_index(drop=True)
    all_preds = []

    # Get to the wrapper so we can call its helper directly
    wrapper = model._model_impl.python_model  

    days_remaining = total_days
    while days_remaining > 0:
        ahead = chunk_size
        # Build context+future dates via your wrapper
        inp = wrapper.make_future_dataframe(history, periods=7, freq='D')
        inp['y'] = inp['y'].fillna(0)  
        # Predict (only needs ds + holidays internally)
        preds = model.predict(inp)

        # Take just the next 'ahead' rows
        last_date = history['ds'].max()
        next_chunk = preds[preds['ds'] > last_date].head(days_remaining)

        # Collect and append
        all_preds.append(next_chunk)
        # Feed back point forecasts as new history
        new_hist = next_chunk[['ds','yhat1']].rename(columns={'yhat1':'y'})
        history = pd.concat([history, new_hist], ignore_index=True)

        days_remaining -= ahead

    # Combine all into one DataFrame
    return pd.concat(all_preds, ignore_index=True)

def recursive_predict(model, history_df, total_days=30, chunk_size=7):
    """
    Corrected recursive prediction that properly handles model input/output
    """
    history = history_df[['ds','y']].copy().sort_values('ds').reset_index(drop=True)
    all_future_dates = []
    all_forecasts = []
    
    wrapper = model.unwrap_python_model()
    
    # Generate the target date range we want to forecast
    start_date = history['ds'].max() + pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=total_days-1)
    target_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"Target forecast range: {start_date.date()} to {end_date.date()} ({total_days} days)")
    
    days_remaining = total_days
    iteration = 0
    
    while days_remaining > 0:
        iteration += 1
        current_chunk = min(chunk_size, days_remaining)
        
        print(f"Iteration {iteration}: Predicting next {current_chunk} days")
        print(f"History currently ends at: {history['ds'].max().date()}")
        
        # Create input with historical context + future dates
        forecast_input = wrapper.make_future_dataframe(
            historical_df=history,
            periods=chunk_size,  # Always use full model horizon
            freq='D'
        )
        
        print(f"  Forecast input range: {forecast_input['ds'].min().date()} to {forecast_input['ds'].max().date()}")
        
        # Make prediction
        preds = model.predict(forecast_input)
        
        # Extract only the future predictions we need for this chunk
        last_hist_date = history['ds'].max()
        future_preds = preds[preds['ds'] > last_hist_date].head(current_chunk).copy()
        
        if len(future_preds) == 0:
            print("No future predictions generated, stopping")
            break
        
        print(f"  Generated predictions for: {future_preds['ds'].min().date()} to {future_preds['ds'].max().date()}")
        
        # Store the actual forecast dates and values we want to keep
        for _, row in future_preds.iterrows():
            if row['ds'] in target_dates:  # Only keep dates in our target range
                all_future_dates.append(row['ds'])
                # Create a clean forecast record using yhat1 (1-step ahead is most reliable)
                forecast_record = {
                    'ds': row['ds'],
                    'yhat1': row.get('yhat1', row.get('yhat', None)),  # Fallback to yhat if yhat1 not available
                    'yhat1 5.0%': row.get('yhat1 5.0%'),
                    'yhat1 95.0%': row.get('yhat1 95.0%'),
                    'trend': row.get('trend'),
                    'season_weekly': row.get('season_weekly'),
                    'event_Holiday': row.get('event_Holiday'),
                }
                all_forecasts.append(forecast_record)
        
        # For next iteration, add the predictions as new history
        # Use yhat1 (1-step-ahead) which is typically most accurate
        new_history_points = []
        for _, row in future_preds.iterrows():
            forecast_value = row.get('yhat1', row.get('yhat', None))
            if forecast_value is not None and pd.notna(forecast_value):
                new_history_points.append({
                    'ds': row['ds'],
                    'y': forecast_value
                })
        
        if new_history_points:
            new_hist_df = pd.DataFrame(new_history_points)
            history = pd.concat([history, new_hist_df], ignore_index=True)
            history = history.sort_values('ds').reset_index(drop=True)
        
        days_remaining -= current_chunk
    
    # Create final forecast dataframe
    if all_forecasts:
        final_df = pd.DataFrame(all_forecasts)
        final_df = final_df.sort_values('ds').reset_index(drop=True)
        print(f"Final forecast: {len(final_df)} days from {final_df['ds'].min().date()} to {final_df['ds'].max().date()}")
        return final_df
    else:
        print("No forecasts generated")
        return pd.DataFrame()

def main():
    scope = 'WellnessSync'
    server = dbutils.secrets.get(scope=scope, key='AZURE_SQL_SERVER')
    database = dbutils.secrets.get(scope=scope, key='AZURE_SQL_DATABASE')
    username = dbutils.secrets.get(scope=scope, key='AZURE_SQL_USERNAME')
    password = dbutils.secrets.get(scope=scope, key='AZURE_SQL_PASSWORD')

    table_name = "forecast"
    schema_name = "DB"
    
    try:

        connector = AzureConnector(server, database, username, password)
        spark = connector.get_spark()

        print("Loading historical data...")
        #historical_df = pd.read_csv('/Workspace/Repos/ryan@delve.systems/Prophet_AI/Wellness_Sales_Grouped.csv')
        historical_df = connector.read_dataframe()
        print("Columns:", historical_df.columns.tolist())
        print("Non-null y count:", historical_df['y'].notna().sum())
        print(historical_df.head())
        
        #historical_df['ds'] = pd.to_datetime(historical_df['ds'])
        #historical_df = historical_df.sort_values('ds').reset_index(drop=True)
        print(f"Historical data: {len(historical_df)} rows.\nDate range: {historical_df['ds'].min()} to {historical_df['ds'].max()}")
        
        # Load holidays
        print("Loading holidays...")
        holidays_df = get_holidays()
        
        # Load model
        print("Loading model...")
        model_name = "WellnessSalesForecast_7_day"
        model_stage = "5"
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")
        
        # Generate forecast
        print("Generating forecast...")
        df_forecast = generate_forecast(
            model=model,
            historical_df=historical_df,
            #holidays_df=holidays_df,
            forecast_periods=7
        )

        #df_forecast = recursive_predict(model, historical_df)
        # print(f"Forecast generated: {len(df_forecast)} rows")
        # print(df_forecast.head())
        flat_df_forecast = flatten_forecast(df_forecast)
        #Convert to Spark DataFrame
        df_forecast_spark = spark.createDataFrame(flat_df_forecast)
        df_forecast_spark.display()
        
        # Write to Azure SQL
        print("Writing forecast to Azure SQL...")


        result = connector.write_dataframe(
            df=df_forecast_spark,
            table_name=table_name,
            schema_name=schema_name
        )
        
        print(f"Write operation completed: {result}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()