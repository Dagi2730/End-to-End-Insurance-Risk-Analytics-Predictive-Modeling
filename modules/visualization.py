"""
AlphaCare Insurance Solutions - Visualization Module

This module contains all visualization functions for the insurance analytics project.
Provides comprehensive plotting capabilities for EDA, statistical analysis, and model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style preferences
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class InsuranceVisualizer:
    """
    Comprehensive visualization class for AlphaCare Insurance Analytics.
    
    Provides static and interactive visualizations for:
    - Exploratory Data Analysis
    - Statistical test results
    - Model performance metrics
    - Business insights
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_data_overview(self, df: pd.DataFrame) -> None:
        """
        Create comprehensive data overview visualizations.
        
        Args:
            df (pd.DataFrame): Dataset to visualize
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AlphaCare Insurance Data Overview', fontsize=16, fontweight='bold')
        
        # Missing values heatmap
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            axes[0, 0].bar(range(len(missing_data)), missing_data.values)
            axes[0, 0].set_xticks(range(len(missing_data)))
            axes[0, 0].set_xticklabels(missing_data.index, rotation=45, ha='right')
            axes[0, 0].set_title('Missing Values by Column')
            axes[0, 0].set_ylabel('Count')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Missing Values by Column')
        
        # Data types distribution
        dtype_counts = df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Data Types Distribution')
        
        # Numerical columns distribution
        numerical_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Top 4
        for i, col in enumerate(numerical_cols):
            if i < 2:
                axes[1, i].hist(df[col].dropna(), bins=30, alpha=0.7, color=self.color_palette[i])
                axes[1, i].set_title(f'Distribution of {col}')
                axes[1, i].set_xlabel(col)
                axes[1, i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def plot_univariate_analysis(self, df: pd.DataFrame, columns: List[str] = None) -> None:
        """
        Create univariate analysis plots for specified columns.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            columns (List[str]): Columns to analyze. If None, analyze key columns.
        """
        if columns is None:
            # Default key columns for insurance analysis
            numerical_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
            categorical_cols = ['Province', 'Gender', 'VehicleType']
            columns = [col for col in numerical_cols + categorical_cols if col in df.columns]
        
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
        fig.suptitle('Univariate Analysis - AlphaCare Insurance Data', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(columns):
            row, col_idx = divmod(i, 3)
            ax = axes[row, col_idx]
            
            if df[col].dtype in ['object', 'category']:
                # Categorical variable - bar plot
                value_counts = df[col].value_counts()
                ax.bar(range(len(value_counts)), value_counts.values, color=self.color_palette[i % len(self.color_palette)])
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_title(f'{col} Distribution')
                ax.set_ylabel('Count')
                
                # Add value labels on bars
                for j, v in enumerate(value_counts.values):
                    ax.text(j, v + max(value_counts.values) * 0.01, str(v), ha='center', va='bottom')
                    
            else:
                # Numerical variable - histogram with KDE
                ax.hist(df[col].dropna(), bins=30, alpha=0.7, density=True, 
                       color=self.color_palette[i % len(self.color_palette)])
                
                # Add KDE curve
                try:
                    from scipy import stats
                    x_range = np.linspace(df[col].min(), df[col].max(), 100)
                    kde = stats.gaussian_kde(df[col].dropna())
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                    ax.legend()
                except:
                    pass
                
                ax.set_title(f'{col} Distribution')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
        
        # Hide empty subplots
        for i in range(len(columns), n_rows * 3):
            row, col_idx = divmod(i, 3)
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> None:
        """
        Create correlation matrix heatmap for numerical variables.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
        """
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            print("No numerical columns found for correlation analysis.")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr(method=method)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        
        plt.title(f'Correlation Matrix ({method.capitalize()})\nAlphaCare Insurance Data', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Print strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if strong_corr:
            print("\n🔍 Strong Correlations (|r| > 0.5):")
            for var1, var2, corr_val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                print(f"   {var1} ↔ {var2}: {corr_val:.3f}")
    
    def plot_bivariate_analysis(self, df: pd.DataFrame, x_col: str, y_col: str, 
                               hue_col: str = None, plot_type: str = 'scatter') -> None:
        """
        Create bivariate analysis plots.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            x_col (str): X-axis variable
            y_col (str): Y-axis variable
            hue_col (str): Variable for color coding
            plot_type (str): Type of plot ('scatter', 'box', 'violin')
        """
        plt.figure(figsize=self.figsize)
        
        if plot_type == 'scatter':
            if hue_col and hue_col in df.columns:
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.6)
            else:
                sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6)
                
            # Add trend line
            try:
                sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color='red', ax=plt.gca())
            except:
                pass
                
        elif plot_type == 'box':
            sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
            plt.xticks(rotation=45)
            
        elif plot_type == 'violin':
            sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col)
            plt.xticks(rotation=45)
        
        plt.title(f'{y_col} vs {x_col}' + (f' by {hue_col}' if hue_col else ''))
        plt.tight_layout()
        plt.show()
    
    def plot_temporal_analysis(self, df: pd.DataFrame, date_col: str, 
                              value_cols: List[str], freq: str = 'M') -> None:
        """
        Create temporal analysis plots for time series data.
        
        Args:
            df (pd.DataFrame): Dataset with temporal data
            date_col (str): Date column name
            value_cols (List[str]): Value columns to plot over time
            freq (str): Frequency for aggregation ('D', 'W', 'M', 'Q', 'Y')
        """
        if date_col not in df.columns:
            print(f"Date column '{date_col}' not found in dataset.")
            return
        
        # Convert to datetime if needed
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
        
        # Set date as index for resampling
        df_temp.set_index(date_col, inplace=True)
        
        # Create subplots
        n_plots = len(value_cols)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle('Temporal Analysis - AlphaCare Insurance Data', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(value_cols):
            if col not in df_temp.columns:
                continue
                
            # Resample data
            resampled_data = df_temp[col].resample(freq).agg(['mean', 'sum', 'count'])
            
            # Plot mean values
            axes[i].plot(resampled_data.index, resampled_data['mean'], 
                        marker='o', linewidth=2, markersize=4, 
                        color=self.color_palette[i % len(self.color_palette)])
            
            axes[i].set_title(f'{col} - Monthly Trends (Mean Values)')
            axes[i].set_ylabel(f'Average {col}')
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line
            try:
                z = np.polyfit(range(len(resampled_data)), resampled_data['mean'], 1)
                p = np.poly1d(z)
                axes[i].plot(resampled_data.index, p(range(len(resampled_data))), 
                           "r--", alpha=0.8, linewidth=1, label='Trend')
                axes[i].legend()
            except:
                pass
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard_plots(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Create interactive Plotly visualizations for the Streamlit dashboard.
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            
        Returns:
            Dict[str, go.Figure]: Dictionary of Plotly figures
        """
        plots = {}
        
        # 1. Premium vs Claims Scatter Plot
        if all(col in df.columns for col in ['TotalPremium', 'TotalClaims']):
            fig_scatter = px.scatter(
                df, x='TotalPremium', y='TotalClaims',
                color='Province' if 'Province' in df.columns else None,
                size='CustomValueEstimate' if 'CustomValueEstimate' in df.columns else None,
                hover_data=['VehicleType'] if 'VehicleType' in df.columns else None,
                title='Premium vs Claims Analysis',
                labels={'TotalPremium': 'Total Premium (ZAR)', 'TotalClaims': 'Total Claims (ZAR)'}
            )
            fig_scatter.update_layout(height=500)
            plots['premium_claims_scatter'] = fig_scatter
        
        # 2. Province-wise Analysis
        if 'Province' in df.columns and 'TotalPremium' in df.columns:
            province_stats = df.groupby('Province').agg({
                'TotalPremium': ['mean', 'sum', 'count'],
                'TotalClaims': ['mean', 'sum'] if 'TotalClaims' in df.columns else ['count']
            }).round(2)
            
            fig_province = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Average Premium by Province', 'Total Claims by Province'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            provinces = province_stats.index
            avg_premium = province_stats[('TotalPremium', 'mean')]
            total_claims = province_stats[('TotalClaims', 'sum')] if 'TotalClaims' in df.columns else [0] * len(provinces)
            
            fig_province.add_trace(
                go.Bar(x=provinces, y=avg_premium, name='Avg Premium', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig_province.add_trace(
                go.Bar(x=provinces, y=total_claims, name='Total Claims', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig_province.update_layout(height=400, title_text="Provincial Analysis")
            plots['province_analysis'] = fig_province
        
        # 3. Vehicle Type Distribution
        if 'VehicleType' in df.columns:
            vehicle_counts = df['VehicleType'].value_counts()
            
            fig_vehicle = go.Figure(data=[
                go.Pie(labels=vehicle_counts.index, values=vehicle_counts.values, hole=0.3)
            ])
            fig_vehicle.update_layout(
                title="Vehicle Type Distribution",
                height=400
            )
            plots['vehicle_distribution'] = fig_vehicle
        
        # 4. Time Series Analysis
        if 'TransactionMonth' in df.columns and 'TotalPremium' in df.columns:
            df_temp = df.copy()
            df_temp['TransactionMonth'] = pd.to_datetime(df_temp['TransactionMonth'])
            
            monthly_data = df_temp.groupby(df_temp['TransactionMonth'].dt.to_period('M')).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum' if 'TotalClaims' in df.columns else 'count'
            }).reset_index()
            
            monthly_data['TransactionMonth'] = monthly_data['TransactionMonth'].astype(str)
            
            fig_time = go.Figure()
            
            fig_time.add_trace(go.Scatter(
                x=monthly_data['TransactionMonth'],
                y=monthly_data['TotalPremium'],
                mode='lines+markers',
                name='Total Premium',
                line=dict(color='blue', width=2)
            ))
            
            if 'TotalClaims' in df.columns:
                fig_time.add_trace(go.Scatter(
                    x=monthly_data['TransactionMonth'],
                    y=monthly_data['TotalClaims'],
                    mode='lines+markers',
                    name='Total Claims',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
            
            fig_time.update_layout(
                title='Monthly Premium and Claims Trends',
                xaxis_title='Month',
                yaxis_title='Total Premium (ZAR)',
                yaxis2=dict(title='Total Claims (ZAR)', overlaying='y', side='right'),
                height=500
            )
            plots['time_series'] = fig_time
        
        return plots
    
    def plot_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              model_name: str = "Model") -> None:
        """
        Create model performance visualization plots.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # 3. Residuals distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', density=True)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residuals Distribution')
        
        # Add normal curve
        try:
            from scipy import stats
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            normal_curve = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
            axes[1, 0].plot(x_range, normal_curve, 'r-', linewidth=2, label='Normal')
            axes[1, 0].legend()
        except:
            pass
        
        # 4. Error metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics_text = f"""
        Performance Metrics:
        
        R² Score: {r2:.4f}
        RMSE: {rmse:.2f}
        MAE: {mae:.2f}
        MSE: {mse:.2f}
        
        Mean Actual: {y_true.mean():.2f}
        Mean Predicted: {y_pred.mean():.2f}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Metrics')
        
        plt.tight_layout()
        plt.show()


# Utility functions for quick plotting
def quick_eda_plots(df: pd.DataFrame) -> None:
    """
    Generate quick EDA plots for a dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    visualizer = InsuranceVisualizer()
    
    print("📊 Generating EDA visualizations...")
    visualizer.plot_data_overview(df)
    visualizer.plot_univariate_analysis(df)
    visualizer.plot_correlation_matrix(df)
    
    # Bivariate analysis for key variables
    if all(col in df.columns for col in ['TotalPremium', 'TotalClaims']):
        visualizer.plot_bivariate_analysis(df, 'TotalPremium', 'TotalClaims', 
                                         hue_col='Province' if 'Province' in df.columns else None)
    
    print("✅ EDA visualizations completed!")


if __name__ == "__main__":
    # Example usage with sample data
    from data_processing import load_sample_data
    
    sample_df = load_sample_data()
    quick_eda_plots(sample_df)
