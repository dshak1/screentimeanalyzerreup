"""
Screen Time Analytics Dashboard
===============================

A comprehensive data analysis tool for mobile app usage patterns.
Analyzes screen time data to identify usage trends, notification effectiveness,
and behavioral patterns across different applications and time periods.

Enhanced with performance optimization, anomaly detection, and predictive analytics.

Author: [Your Name]
Created: 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PERFORMANCE METRICS TRACKING
# ==========================================
class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            'data_loading_time': 0,
            'data_processing_time': 0,
            'anomaly_detection_time': 0,
            'forecasting_time': 0,
            'total_records_processed': 0,
            'data_quality_score': 0,
            'anomalies_detected': 0,
            'processing_throughput': 0
        }
    
    def start_timer(self):
        return time.time()
    
    def record_time(self, start_time, metric_name):
        elapsed = time.time() - start_time
        self.metrics[metric_name] = elapsed
        return elapsed
    
    def get_metrics(self):
        return self.metrics.copy()

# ==========================================
# DATA VALIDATION & QUALITY METRICS
# ==========================================
def validate_and_quality_check(data):
    """Perform data validation and calculate quality metrics"""
    quality_metrics = {
        'total_records': len(data),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_records': data.duplicated().sum(),
        'data_completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
        'date_range_days': (pd.to_datetime(data['Date']).max() - pd.to_datetime(data['Date']).min()).days,
        'unique_apps': data['App'].nunique(),
        'valid_usage_records': (data['Usage (minutes)'] >= 0).sum(),
        'valid_notification_records': (data['Notifications'] >= 0).sum()
    }
    
    # Calculate overall quality score
    quality_score = (
        quality_metrics['data_completeness'] * 0.4 +
        (quality_metrics['valid_usage_records'] / quality_metrics['total_records'] * 100) * 0.3 +
        (quality_metrics['valid_notification_records'] / quality_metrics['total_records'] * 100) * 0.3
    )
    quality_metrics['quality_score'] = round(quality_score, 2)
    
    return quality_metrics

# ==========================================
# ANOMALY DETECTION USING IQR AND Z-SCORE METHODS
# ==========================================
def detect_anomalies(data, app_col='App', usage_col='Usage (minutes)', iqr_threshold=1.5, zscore_threshold=2.5):
    """
    Detect anomalies using both Interquartile Range (IQR) and Z-score methods
    Returns anomalies and statistics
    """
    anomalies_iqr = []
    anomalies_zscore = []
    anomaly_stats = {}
    
    for app in data[app_col].unique():
        app_data_full = data[data[app_col] == app]
        app_data = app_data_full[usage_col]
        
        if len(app_data) < 3:
            continue
        
        app_anomalies_iqr = pd.DataFrame()
        app_anomalies_zscore = pd.DataFrame()
        
        # Method 1: IQR-based detection
        Q1 = app_data.quantile(0.25)
        Q3 = app_data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound_iqr = Q1 - iqr_threshold * IQR
            upper_bound_iqr = Q3 + iqr_threshold * IQR
            
            app_anomalies_iqr = app_data_full[(app_data < lower_bound_iqr) | (app_data > upper_bound_iqr)]
        
        # Method 2: Z-score based detection
        mean = app_data.mean()
        std = app_data.std()
        
        if std > 0:
            z_scores = np.abs((app_data - mean) / std)
            app_anomalies_zscore = app_data_full[z_scores > zscore_threshold]
        
        # Combine both methods (union of anomalies)
        all_app_anomalies_list = []
        if len(app_anomalies_iqr) > 0:
            all_app_anomalies_list.append(app_anomalies_iqr)
            anomalies_iqr.append(app_anomalies_iqr)
        if len(app_anomalies_zscore) > 0:
            all_app_anomalies_list.append(app_anomalies_zscore)
            anomalies_zscore.append(app_anomalies_zscore)
        
        if all_app_anomalies_list:
            all_app_anomalies = pd.concat(all_app_anomalies_list).drop_duplicates()
            
            anomaly_stats[app] = {
                'count': len(all_app_anomalies),
                'iqr_count': len(app_anomalies_iqr),
                'zscore_count': len(app_anomalies_zscore),
                'percentage': round(len(all_app_anomalies) / len(app_data_full) * 100, 2),
                'mean': round(mean, 2),
                'std': round(std, 2)
            }
    
    # Combine all anomalies from all apps
    all_anomalies_list = []
    if anomalies_iqr:
        all_anomalies_list.extend(anomalies_iqr)
    if anomalies_zscore:
        all_anomalies_list.extend(anomalies_zscore)
    
    if all_anomalies_list:
        all_anomalies = pd.concat(all_anomalies_list).drop_duplicates().reset_index(drop=True)
    else:
        all_anomalies = pd.DataFrame()
    
    return all_anomalies, anomaly_stats

# ==========================================
# TIME-SERIES FORECASTING
# ==========================================
def forecast_usage(data, app_name, days_ahead=7):
    """
    Forecast future app usage using linear regression
    Returns forecasted values and model performance metrics
    """
    app_data = data[data['App'] == app_name].copy()
    if len(app_data) < 5:
        return None, None
    
    app_data = app_data.sort_values('Date')
    app_data['Day_Number'] = (app_data['Date'] - app_data['Date'].min()).dt.days
    
    X = app_data[['Day_Number']].values
    y = app_data['Usage (minutes)'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R² score
    y_pred = model.predict(X)
    r2_score = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    
    # Forecast future days
    last_day = app_data['Day_Number'].max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
    forecast = model.predict(future_days)
    
    # Create forecast dataframe
    last_date = app_data['Date'].max()
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'App': app_name,
        'Forecasted_Usage': forecast,
        'Model_R2': r2_score
    })
    
    return forecast_df, {
        'r2_score': round(r2_score, 4),
        'model_coefficient': round(model.coef_[0], 4),
        'forecast_days': days_ahead
    }

# ==========================================
# OPTIMIZED DATA PROCESSING PIPELINE
# ==========================================
def process_data_optimized(data_path, perf_metrics):
    """Optimized data processing pipeline with vectorized operations"""
    start_time = perf_metrics.start_timer()
    
    # Load data
    data = pd.read_csv(data_path)
    load_time = perf_metrics.record_time(start_time, 'data_loading_time')
    
    # Vectorized preprocessing
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day of Week'] = data['Date'].dt.day_name()
    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week
    
    # Calculate processing throughput
    processing_time = perf_metrics.record_time(start_time, 'data_processing_time')
    throughput = len(data) / processing_time if processing_time > 0 else 0
    perf_metrics.metrics['processing_throughput'] = round(throughput, 2)
    perf_metrics.metrics['total_records_processed'] = len(data)
    
    print(f"✓ Data loaded in {load_time:.3f}s")
    print(f"✓ Processed {len(data)} records at {throughput:.0f} records/second")
    
    return data

# ==========================================
# GENERATE COMPREHENSIVE REPORT
# ==========================================
def generate_report(data, quality_metrics, anomaly_stats, forecasts, perf_metrics, output_dir='reports'):
    """Generate comprehensive analytics report with metrics"""
    Path(output_dir).mkdir(exist_ok=True)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'data_quality': quality_metrics,
        'performance_metrics': perf_metrics.get_metrics(),
        'anomaly_detection': {
            'total_anomalies': sum(stats['count'] for stats in anomaly_stats.values()),
            'apps_with_anomalies': len(anomaly_stats),
            'anomaly_details': anomaly_stats
        },
        'forecasting_summary': {
            'apps_forecasted': len([f for f in forecasts if f is not None]),
            'forecast_period_days': 7
        },
        'summary_statistics': {
            'total_records': len(data),
            'unique_apps': data['App'].nunique(),
            'date_range': {
                'start': str(data['Date'].min()),
                'end': str(data['Date'].max()),
                'days': quality_metrics['date_range_days']
            },
            'average_daily_usage': round(data['Usage (minutes)'].mean(), 2),
            'total_usage_hours': round(data['Usage (minutes)'].sum() / 60, 2)
        }
    }
    
    # Save JSON report (convert numpy types to native Python types)
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    report_serializable = convert_to_native(report)
    report_path = Path(output_dir) / f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report_serializable, f, indent=2)
    
    # Save CSV summary
    summary_df = pd.DataFrame([{
        'Metric': 'Total Records',
        'Value': report['summary_statistics']['total_records']
    }, {
        'Metric': 'Unique Apps',
        'Value': report['summary_statistics']['unique_apps']
    }, {
        'Metric': 'Data Quality Score',
        'Value': quality_metrics['quality_score']
    }, {
        'Metric': 'Processing Throughput (records/sec)',
        'Value': perf_metrics.metrics['processing_throughput']
    }, {
        'Metric': 'Anomalies Detected',
        'Value': report['anomaly_detection']['total_anomalies']
    }])
    
    csv_path = Path(output_dir) / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Report generated: {report_path}")
    print(f"✓ Summary CSV: {csv_path}")
    
    return report

# ==========================================
# MAIN ANALYSIS WORKFLOW
# ==========================================
def main(data_path='content/screentime_analysis.csv', generate_plots=True, export_reports=True):
    """Main analysis workflow with enhanced features"""
    perf_metrics = PerformanceMetrics()
    
    print("=" * 60)
    print("Screen Time Analytics Dashboard - Enhanced Version")
    print("=" * 60)
    
    # Step 1: Load and process data
    print("\n[1/5] Loading and processing data...")
    data = process_data_optimized(data_path, perf_metrics)
    
    # Step 2: Data validation
    print("\n[2/5] Validating data quality...")
    quality_metrics = validate_and_quality_check(data)
    perf_metrics.metrics['data_quality_score'] = quality_metrics['quality_score']
    
    print(f"✓ Data Quality Score: {quality_metrics['quality_score']}%")
    print(f"✓ Records: {quality_metrics['total_records']}, Apps: {quality_metrics['unique_apps']}")
    print(f"✓ Date Range: {quality_metrics['date_range_days']} days")
    
    # Step 3: Anomaly detection
    print("\n[3/5] Detecting anomalies...")
    start_time = perf_metrics.start_timer()
    anomalies, anomaly_stats = detect_anomalies(data)
    perf_metrics.record_time(start_time, 'anomaly_detection_time')
    perf_metrics.metrics['anomalies_detected'] = len(anomalies)
    
    total_anomalies = sum(stats['count'] for stats in anomaly_stats.values())
    print(f"✓ Detected {total_anomalies} anomalies across {len(anomaly_stats)} apps")
    
    # Step 4: Forecasting
    print("\n[4/5] Generating usage forecasts...")
    start_time = perf_metrics.start_timer()
    forecasts = []
    forecast_models = {}
    
    top_apps = data.groupby('App')['Usage (minutes)'].sum().nlargest(5).index
    for app in top_apps:
        forecast_df, model_metrics = forecast_usage(data, app, days_ahead=7)
        if forecast_df is not None:
            forecasts.append(forecast_df)
            forecast_models[app] = model_metrics
    
    perf_metrics.record_time(start_time, 'forecasting_time')
    print(f"✓ Generated forecasts for {len(forecast_models)} apps")
    for app, metrics in forecast_models.items():
        print(f"  - {app}: R² = {metrics['r2_score']:.3f}")
    
    # Step 5: Generate visualizations (if requested)
    if generate_plots:
        print("\n[5/5] Generating visualizations...")
        generate_visualizations(data, anomalies, forecasts)
    
    # Step 6: Export reports (if requested)
    if export_reports:
        print("\n[6/6] Generating reports...")
        report = generate_report(data, quality_metrics, anomaly_stats, forecasts, perf_metrics)
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYTICS SUMMARY")
        print("=" * 60)
        print(f"Total Records Processed: {perf_metrics.metrics['total_records_processed']}")
        print(f"Processing Throughput: {perf_metrics.metrics['processing_throughput']:.0f} records/sec")
        print(f"Data Quality Score: {perf_metrics.metrics['data_quality_score']}%")
        print(f"Anomalies Detected: {perf_metrics.metrics['anomalies_detected']}")
        print(f"Total Processing Time: {sum([perf_metrics.metrics[k] for k in ['data_loading_time', 'data_processing_time', 'anomaly_detection_time', 'forecasting_time']]):.3f}s")
        print("=" * 60)
    
    return data, quality_metrics, anomaly_stats, forecasts, perf_metrics

# ==========================================
# VISUALIZATION GENERATION
# ==========================================
def generate_visualizations(data, anomalies, forecasts):
    """Generate comprehensive visualizations"""
    
    # Time series with anomaly highlighting
    plt.figure(figsize=(14, 6))
    for app in data['App'].unique()[:5]:  # Top 5 apps
        app_data = data[data['App'] == app]
        plt.plot(app_data['Date'], app_data['Usage (minutes)'], 
                marker='o', label=app, alpha=0.7, linewidth=2)
    
    # Highlight anomalies
    if len(anomalies) > 0:
        plt.scatter(anomalies['Date'], anomalies['Usage (minutes)'], 
                   color='red', marker='x', s=100, label='Anomalies', zorder=5)
    
    plt.title('Screen Time Trends with Anomaly Detection', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Usage (minutes)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/time_series_with_anomalies.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Forecast visualization
    if forecasts:
        plt.figure(figsize=(14, 6))
        for forecast_df in forecasts[:3]:  # Show top 3 forecasts
            app_name = forecast_df['App'].iloc[0]
            historical = data[data['App'] == app_name].sort_values('Date')
            
            plt.plot(historical['Date'], historical['Usage (minutes)'], 
                    marker='o', label=f'{app_name} (Historical)', linewidth=2)
            plt.plot(forecast_df['Date'], forecast_df['Forecasted_Usage'], 
                    '--', marker='s', label=f'{app_name} (Forecast)', linewidth=2, alpha=0.7)
        
        plt.title('Usage Forecasting: Historical vs Predicted', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Usage (minutes)', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/forecast_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[['Usage (minutes)', 'Notifications', 'Times Opened']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Usage, Notifications, and App Opens', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Screen Time Analytics Dashboard')
    parser.add_argument('--data', type=str, default='content/screentime_analysis.csv',
                       help='Path to CSV data file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip report generation')
    
    args = parser.parse_args()
    
    main(data_path=args.data, 
         generate_plots=not args.no_plots, 
         export_reports=not args.no_reports)
