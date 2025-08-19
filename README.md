# AlphaCare Insurance Solutions (ACIS) - Analytics Project

## 🏢 Project Overview

This project provides a comprehensive analysis of historical car insurance claims data for AlphaCare Insurance Solutions in South Africa (February 2014 - August 2015). The analysis includes exploratory data analysis, statistical hypothesis testing, predictive modeling, and an interactive dashboard for business insights.

## 📊 Key Features

- **Comprehensive EDA**: Deep dive into insurance claims patterns and trends
- **Statistical Testing**: Hypothesis testing for risk assessment across different segments
- **Predictive Modeling**: Multiple ML models for premium prediction and risk assessment
- **Interactive Dashboard**: Streamlit-based dashboard for real-time insights
- **Model Interpretability**: SHAP analysis for understanding model decisions
- **Production Ready**: Modular, documented, and deployable codebase

## 🏗️ Project Structure

\`\`\`
ACIS-Insurance-Analytics/
├── data/
│   ├── insurance_data.csv          # Raw insurance dataset
│   └── processed/                  # Processed data files
├── notebooks/
│   ├── 1_EDA.ipynb                # Exploratory Data Analysis
│   ├── 2_Hypothesis_Testing.ipynb  # Statistical hypothesis testing
│   └── 3_Predictive_Modeling.ipynb # Machine learning models
├── modules/
│   ├── data_processing.py          # Data preprocessing utilities
│   ├── stats_tests.py             # Statistical testing functions
│   ├── modeling_utils.py          # ML modeling pipeline
│   └── visualization.py           # Visualization utilities
├── dashboard/
│   └── app.py                     # Streamlit dashboard application
├── models/                        # Saved model artifacts
├── outputs/                       # Analysis outputs and reports
├── scripts/                       # Utility scripts
├── dvc.yaml                       # DVC pipeline configuration
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
\`\`\`

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git (for version control)
- DVC (for data versioning)

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd ACIS-Insurance-Analytics
   \`\`\`

2. **Create virtual environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Initialize DVC (optional)**
   \`\`\`bash
   dvc init
   dvc remote add -d localstorage /path/to/your/storage
   \`\`\`

### Running the Analysis

1. **Data Processing**
   ```python
   from modules.data_processing import InsuranceDataProcessor
   
   processor = InsuranceDataProcessor()
   processed_data = processor.run_full_preprocessing()
   \`\`\`

2. **Exploratory Data Analysis**
   \`\`\`bash
   jupyter notebook notebooks/1_EDA.ipynb
   \`\`\`

3. **Statistical Testing**
   \`\`\`bash
   jupyter notebook notebooks/2_Hypothesis_Testing.ipynb
   \`\`\`

4. **Predictive Modeling**
   \`\`\`bash
   jupyter notebook notebooks/3_Predictive_Modeling.ipynb
   \`\`\`

5. **Launch Dashboard**
   \`\`\`bash
   streamlit run dashboard/app.py
   \`\`\`

## 📈 Analysis Components

### 1. Exploratory Data Analysis (EDA)

- **Data Quality Assessment**: Missing values, duplicates, data types
- **Descriptive Statistics**: Summary statistics for key variables
- **Univariate Analysis**: Distribution analysis for all variables
- **Bivariate Analysis**: Relationships between variables
- **Temporal Analysis**: Time-based trends and patterns
- **Geographic Analysis**: Provincial and zip code insights

**Key Insights:**
- Premium and claims distribution patterns
- Geographic risk variations
- Temporal trends in insurance metrics
- Vehicle type and customer demographic patterns

### 2. Statistical Hypothesis Testing

The project tests four main hypotheses:

#### H₁: Risk Differences Across Provinces
- **Null Hypothesis**: No risk differences across provinces
- **Tests**: Chi-square test for claim frequency, ANOVA for claim severity
- **Metrics**: Claim frequency, claim severity, profit margins

#### H₂: Risk Differences Across Zip Codes
- **Null Hypothesis**: No risk differences across zip codes
- **Tests**: Chi-square test, ANOVA (sample-based due to high cardinality)
- **Approach**: Focus on top zip codes by volume

#### H₃: Margin Differences Between Zip Codes
- **Null Hypothesis**: No significant margin differences between zip codes
- **Tests**: ANOVA, post-hoc analysis
- **Business Impact**: Pricing strategy optimization

#### H₄: Gender Risk Differences
- **Null Hypothesis**: No significant risk differences between women and men
- **Tests**: Chi-square test, independent t-tests
- **Metrics**: Claim frequency, severity, and profitability by gender

### 3. Predictive Modeling

#### Models Implemented:
1. **Linear Regression**: Baseline interpretable model
2. **Random Forest**: Ensemble method for non-linear patterns
3. **XGBoost**: Gradient boosting for optimal performance

#### Model Evaluation:
- **Metrics**: RMSE, MAE, R² score
- **Validation**: Train/validation/test split with cross-validation
- **Feature Importance**: Built-in importance + SHAP analysis
- **Model Comparison**: Comprehensive performance comparison

#### Feature Engineering:
- **Financial Ratios**: Claims ratio, profit margin, premium-to-value ratio
- **Temporal Features**: Vehicle age, customer tenure
- **Risk Categories**: Low/medium/high risk segmentation
- **Geographic Encoding**: Province and zip code features

### 4. Model Interpretability

#### SHAP (SHapley Additive exPlanations) Analysis:
- **Global Importance**: Overall feature impact across all predictions
- **Local Explanations**: Individual prediction explanations
- **Feature Interactions**: Understanding complex relationships
- **Business Insights**: Actionable insights for pricing and risk assessment

## 🎯 Business Applications

### Risk Assessment
- Identify high-risk customer segments
- Geographic risk mapping
- Premium pricing optimization
- Claims prediction and prevention

### Marketing Strategy
- Customer segmentation for targeted campaigns
- Geographic expansion opportunities
- Product customization by demographics
- Retention strategy development

### Operational Efficiency
- Claims processing optimization
- Resource allocation by region
- Fraud detection indicators
- Performance monitoring dashboards

## 📊 Dashboard Features

The Streamlit dashboard provides:

### Interactive Visualizations
- **Premium vs Claims Analysis**: Scatter plots with filtering
- **Geographic Insights**: Provincial and zip code comparisons
- **Temporal Trends**: Time series analysis with drill-down
- **Risk Segmentation**: Customer risk profiling

### Model Predictions
- **Premium Calculator**: Input customer/vehicle details for premium prediction
- **Risk Assessment**: Real-time risk scoring
- **What-If Analysis**: Scenario modeling for business decisions

### Statistical Results
- **Hypothesis Test Results**: Interactive tables and visualizations
- **Model Performance**: Comparison metrics and validation results
- **Feature Importance**: SHAP-based explanations

## 🔧 Technical Implementation

### Data Processing Pipeline
```python
# Example usage
from modules.data_processing import InsuranceDataProcessor

processor = InsuranceDataProcessor("data/insurance_data.csv")
processed_data = processor.run_full_preprocessing()
