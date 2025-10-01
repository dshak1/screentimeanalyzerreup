# Screen Time Analytics Dashboard

A comprehensive data analysis project that leverages Python and statistical libraries to analyze mobile app usage patterns, providing actionable insights into digital behavior and screen time trends.

## 🎯 Project Overview

This analytics tool processes screen time data to identify usage patterns, analyze notification effectiveness, and provide data-driven insights into digital consumption habits. The project includes both CSV-based analysis and direct macOS system integration for real-time data extraction.

## 🔧 Technical Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Advanced data visualization
- **SQLite3**: macOS system database integration
- **Statistical Analysis**: Correlation analysis, trend identification

## 📊 Key Features

### Data Analysis Capabilities
- **Multi-dimensional Analysis**: Processes usage time, notification frequency, and app opening patterns
- **Temporal Trend Analysis**: Identifies weekly usage patterns and day-of-week variations
- **Cross-platform Compatibility**: Supports both CSV import and direct macOS system integration
- **Statistical Correlation**: Analyzes relationships between notifications and user engagement

### Visualization Dashboard
- **Time Series Plotting**: Track usage trends across multiple applications
- **Comparative Analysis**: Side-by-side app usage comparisons
- **Weekly Pattern Recognition**: Visualizes usage patterns by day of week
- **Multi-variable Relationships**: Scatter plot matrices showing correlations

### Advanced Analytics
- **Behavioral Insights**: Calculates probability of app opening based on notifications
- **Usage Aggregation**: Provides mean, median, and distribution statistics
- **Top App Analysis**: Focuses on high-usage applications (Instagram, Netflix, WhatsApp)

## 🚀 Usage

### Basic Analysis (CSV-based)
```bash
python sca.py
```

### macOS System Integration
```bash
python sta_mac.py
```

## 📈 Output Examples

The tool generates comprehensive visualizations including:
- Time series plots of app usage trends
- Weekly usage pattern analysis
- Correlation matrices for usage metrics
- Comparative bar charts for top applications

## 🔮 Technical Achievements

- **Data Pipeline**: Automated data processing from raw screen time metrics
- **System Integration**: Direct database queries to macOS Knowledge framework
- **Statistical Modeling**: Implementation of probability calculations for user behavior
- **Visualization Excellence**: Production-quality charts with custom styling and annotations

## 📁 Project Structure

```
screen_time_analyzer/
├── sca.py                    # Core analysis script
├── sta_mac.py               # macOS system integration
├── content/
│   └── screentime_analysis.csv  # Sample dataset
└── README.md
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/screen_time_analyzer.git

# Install dependencies
pip install pandas matplotlib seaborn sqlite3
```

## 📊 Data Schema

The analysis works with screen time data containing:
- **Date**: Timestamp of usage
- **App**: Application name
- **Usage (minutes)**: Time spent in application
- **Notifications**: Number of notifications received
- **Times Opened**: Frequency of app launches