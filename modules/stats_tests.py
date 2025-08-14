"""
AlphaCare Insurance Solutions - Statistical Testing Module

This module contains all statistical testing functions for hypothesis testing
in the insurance analytics project. Provides comprehensive statistical analysis
capabilities for business decision making.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, levene
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class InsuranceStatsTester:
    """
    Comprehensive statistical testing class for AlphaCare Insurance Analytics.
    
    Provides statistical tests for:
    - Risk differences across provinces and zip codes
    - Margin differences between groups
    - Gender-based risk analysis
    - A/B testing capabilities
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical tester.
        
        Args:
            alpha (float): Significance level for hypothesis tests
        """
        self.alpha = alpha
        self.test_results = {}
        
    def calculate_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate key risk metrics for insurance analysis.
        
        Args:
            df (pd.DataFrame): Insurance dataset
            
        Returns:
            pd.DataFrame: Dataset with calculated risk metrics
        """
        df_metrics = df.copy()
        
        # Claim Frequency (binary: has claims or not)
        if 'TotalClaims' in df.columns:
            df_metrics['HasClaims'] = (df_metrics['TotalClaims'] > 0).astype(int)
            df_metrics['ClaimFrequency'] = df_metrics['HasClaims']
        
        # Claim Severity (average claim amount when claims occur)
        if 'TotalClaims' in df.columns:
            df_metrics['ClaimSeverity'] = np.where(
                df_metrics['TotalClaims'] > 0,
                df_metrics['TotalClaims'],
                0
            )
        
        # Margin (Profit/Loss)
        if all(col in df.columns for col in ['TotalPremium', 'TotalClaims']):
            df_metrics['Margin'] = df_metrics['TotalPremium'] - df_metrics['TotalClaims']
            df_metrics['MarginRatio'] = np.where(
                df_metrics['TotalPremium'] > 0,
                df_metrics['Margin'] / df_metrics['TotalPremium'],
                0
            )
        
        # Claims Ratio
        if all(col in df.columns for col in ['TotalPremium', 'TotalClaims']):
            df_metrics['ClaimsRatio'] = np.where(
                df_metrics['TotalPremium'] > 0,
                df_metrics['TotalClaims'] / df_metrics['TotalPremium'],
                0
            )
        
        print(f"✅ Risk metrics calculated for {len(df_metrics)} records")
        return df_metrics
    
    def test_risk_differences_provinces(self, df: pd.DataFrame) -> Dict:
        """
        Test for risk differences across provinces.
        
        Null Hypothesis: No risk differences across provinces
        
        Args:
            df (pd.DataFrame): Dataset with risk metrics
            
        Returns:
            Dict: Test results and statistics
        """
        if 'Province' not in df.columns:
            return {"error": "Province column not found"}
        
        # Calculate risk metrics if not present
        if 'ClaimFrequency' not in df.columns:
            df = self.calculate_risk_metrics(df)
        
        results = {
            'test_name': 'Risk Differences Across Provinces',
            'null_hypothesis': 'No risk differences across provinces',
            'alpha': self.alpha
        }
        
        provinces = df['Province'].unique()
        
        # Test 1: Claim Frequency Differences (Chi-square test)
        if 'ClaimFrequency' in df.columns:
            contingency_table = pd.crosstab(df['Province'], df['ClaimFrequency'])
            chi2_stat, chi2_p_value, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
            
            results['claim_frequency_test'] = {
                'test_type': 'Chi-square test of independence',
                'statistic': chi2_stat,
                'p_value': chi2_p_value,
                'degrees_of_freedom': chi2_dof,
                'reject_null': chi2_p_value < self.alpha,
                'interpretation': 'Significant risk differences across provinces' if chi2_p_value < self.alpha 
                               else 'No significant risk differences across provinces'
            }
        
        # Test 2: Claim Severity Differences (ANOVA)
        if 'ClaimSeverity' in df.columns:
            province_groups = [df[df['Province'] == province]['ClaimSeverity'].dropna() 
                             for province in provinces]
            
            # Remove empty groups
            province_groups = [group for group in province_groups if len(group) > 0]
            
            if len(province_groups) >= 2:
                f_stat, anova_p_value = stats.f_oneway(*province_groups)
                
                results['claim_severity_test'] = {
                    'test_type': 'One-way ANOVA',
                    'statistic': f_stat,
                    'p_value': anova_p_value,
                    'reject_null': anova_p_value < self.alpha,
                    'interpretation': 'Significant claim severity differences across provinces' if anova_p_value < self.alpha 
                                   else 'No significant claim severity differences across provinces'
                }
        
        # Test 3: Margin Differences (ANOVA)
        if 'Margin' in df.columns:
            margin_groups = [df[df['Province'] == province]['Margin'].dropna() 
                           for province in provinces]
            
            # Remove empty groups
            margin_groups = [group for group in margin_groups if len(group) > 0]
            
            if len(margin_groups) >= 2:
                f_stat, anova_p_value = stats.f_oneway(*margin_groups)
                
                results['margin_test'] = {
                    'test_type': 'One-way ANOVA',
                    'statistic': f_stat,
                    'p_value': anova_p_value,
                    'reject_null': anova_p_value < self.alpha,
                    'interpretation': 'Significant margin differences across provinces' if anova_p_value < self.alpha 
                                   else 'No significant margin differences across provinces'
                }
        
        # Summary statistics by province
        summary_stats = df.groupby('Province').agg({
            'ClaimFrequency': ['mean', 'count'] if 'ClaimFrequency' in df.columns else ['count'],
            'ClaimSeverity': ['mean', 'std'] if 'ClaimSeverity' in df.columns else ['count'],
            'Margin': ['mean', 'std'] if 'Margin' in df.columns else ['count']
        }).round(4)
        
        results['summary_statistics'] = summary_stats
        
        self.test_results['province_risk_test'] = results
        return results
    
    def test_risk_differences_zipcodes(self, df: pd.DataFrame, sample_zipcodes: int = 10) -> Dict:
        """
        Test for risk differences across zip codes (sample-based due to high cardinality).
        
        Null Hypothesis: No risk differences across zip codes
        
        Args:
            df (pd.DataFrame): Dataset with risk metrics
            sample_zipcodes (int): Number of top zip codes to sample for testing
            
        Returns:
            Dict: Test results and statistics
        """
        if 'ZipCode' not in df.columns:
            return {"error": "ZipCode column not found"}
        
        # Calculate risk metrics if not present
        if 'ClaimFrequency' not in df.columns:
            df = self.calculate_risk_metrics(df)
        
        # Sample top zip codes by frequency
        top_zipcodes = df['ZipCode'].value_counts().head(sample_zipcodes).index
        df_sample = df[df['ZipCode'].isin(top_zipcodes)]
        
        results = {
            'test_name': 'Risk Differences Across Zip Codes',
            'null_hypothesis': 'No risk differences across zip codes',
            'alpha': self.alpha,
            'sample_size': len(df_sample),
            'zipcodes_tested': list(top_zipcodes)
        }
        
        # Test 1: Claim Frequency Differences (Chi-square test)
        if 'ClaimFrequency' in df_sample.columns:
            contingency_table = pd.crosstab(df_sample['ZipCode'], df_sample['ClaimFrequency'])
            
            # Only proceed if we have sufficient data
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2_stat, chi2_p_value, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
                
                results['claim_frequency_test'] = {
                    'test_type': 'Chi-square test of independence',
                    'statistic': chi2_stat,
                    'p_value': chi2_p_value,
                    'degrees_of_freedom': chi2_dof,
                    'reject_null': chi2_p_value < self.alpha,
                    'interpretation': 'Significant risk differences across zip codes' if chi2_p_value < self.alpha 
                                   else 'No significant risk differences across zip codes'
                }
        
        # Test 2: Margin Differences (ANOVA)
        if 'Margin' in df_sample.columns:
            zipcode_groups = [df_sample[df_sample['ZipCode'] == zipcode]['Margin'].dropna() 
                            for zipcode in top_zipcodes]
            
            # Remove empty groups and groups with insufficient data
            zipcode_groups = [group for group in zipcode_groups if len(group) >= 5]
            
            if len(zipcode_groups) >= 2:
                f_stat, anova_p_value = stats.f_oneway(*zipcode_groups)
                
                results['margin_test'] = {
                    'test_type': 'One-way ANOVA',
                    'statistic': f_stat,
                    'p_value': anova_p_value,
                    'reject_null': anova_p_value < self.alpha,
                    'interpretation': 'Significant margin differences across zip codes' if anova_p_value < self.alpha 
                                   else 'No significant margin differences across zip codes'
                }
        
        # Summary statistics by zip code
        summary_stats = df_sample.groupby('ZipCode').agg({
            'ClaimFrequency': ['mean', 'count'] if 'ClaimFrequency' in df_sample.columns else ['count'],
            'Margin': ['mean', 'std'] if 'Margin' in df_sample.columns else ['count']
        }).round(4)
        
        results['summary_statistics'] = summary_stats
        
        self.test_results['zipcode_risk_test'] = results
        return results
    
    def test_gender_risk_differences(self, df: pd.DataFrame) -> Dict:
        """
        Test for risk differences between genders.
        
        Null Hypothesis: No significant risk differences between women and men
        
        Args:
            df (pd.DataFrame): Dataset with risk metrics
            
        Returns:
            Dict: Test results and statistics
        """
        if 'Gender' not in df.columns:
            return {"error": "Gender column not found"}
        
        # Calculate risk metrics if not present
        if 'ClaimFrequency' not in df.columns:
            df = self.calculate_risk_metrics(df)
        
        results = {
            'test_name': 'Gender Risk Differences',
            'null_hypothesis': 'No significant risk differences between women and men',
            'alpha': self.alpha
        }
        
        # Filter for male and female only
        gender_df = df[df['Gender'].isin(['Male', 'Female', 'M', 'F'])].copy()
        
        if len(gender_df) == 0:
            return {"error": "No valid gender data found"}
        
        # Standardize gender values
        gender_df['Gender'] = gender_df['Gender'].map({
            'Male': 'Male', 'M': 'Male', 'Female': 'Female', 'F': 'Female'
        })
        
        male_data = gender_df[gender_df['Gender'] == 'Male']
        female_data = gender_df[gender_df['Gender'] == 'Female']
        
        # Test 1: Claim Frequency Differences (Chi-square test)
        if 'ClaimFrequency' in gender_df.columns:
            contingency_table = pd.crosstab(gender_df['Gender'], gender_df['ClaimFrequency'])
            chi2_stat, chi2_p_value, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
            
            results['claim_frequency_test'] = {
                'test_type': 'Chi-square test of independence',
                'statistic': chi2_stat,
                'p_value': chi2_p_value,
                'degrees_of_freedom': chi2_dof,
                'reject_null': chi2_p_value < self.alpha,
                'interpretation': 'Significant gender differences in claim frequency' if chi2_p_value < self.alpha 
                               else 'No significant gender differences in claim frequency'
            }
        
        # Test 2: Claim Severity Differences (t-test)
        if 'ClaimSeverity' in gender_df.columns:
            male_severity = male_data['ClaimSeverity'].dropna()
            female_severity = female_data['ClaimSeverity'].dropna()
            
            if len(male_severity) > 0 and len(female_severity) > 0:
                # Check for equal variances
                levene_stat, levene_p = levene(male_severity, female_severity)
                equal_var = levene_p > 0.05
                
                # Perform t-test
                t_stat, t_p_value = ttest_ind(male_severity, female_severity, equal_var=equal_var)
                
                results['claim_severity_test'] = {
                    'test_type': f'Independent t-test (equal_var={equal_var})',
                    'statistic': t_stat,
                    'p_value': t_p_value,
                    'reject_null': t_p_value < self.alpha,
                    'interpretation': 'Significant gender differences in claim severity' if t_p_value < self.alpha 
                                   else 'No significant gender differences in claim severity'
                }
        
        # Test 3: Margin Differences (t-test)
        if 'Margin' in gender_df.columns:
            male_margin = male_data['Margin'].dropna()
            female_margin = female_data['Margin'].dropna()
            
            if len(male_margin) > 0 and len(female_margin) > 0:
                # Check for equal variances
                levene_stat, levene_p = levene(male_margin, female_margin)
                equal_var = levene_p > 0.05
                
                # Perform t-test
                t_stat, t_p_value = ttest_ind(male_margin, female_margin, equal_var=equal_var)
                
                results['margin_test'] = {
                    'test_type': f'Independent t-test (equal_var={equal_var})',
                    'statistic': t_stat,
                    'p_value': t_p_value,
                    'reject_null': t_p_value < self.alpha,
                    'interpretation': 'Significant gender differences in margin' if t_p_value < self.alpha 
                                   else 'No significant gender differences in margin'
                }
        
        # Summary statistics by gender
        summary_stats = gender_df.groupby('Gender').agg({
            'ClaimFrequency': ['mean', 'count'] if 'ClaimFrequency' in gender_df.columns else ['count'],
            'ClaimSeverity': ['mean', 'std'] if 'ClaimSeverity' in gender_df.columns else ['count'],
            'Margin': ['mean', 'std'] if 'Margin' in gender_df.columns else ['count']
        }).round(4)
        
        results['summary_statistics'] = summary_stats
        
        self.test_results['gender_risk_test'] = results
        return results
    
    def perform_ab_test(self, df: pd.DataFrame, group_col: str, metric_col: str, 
                       control_group: str, test_group: str) -> Dict:
        """
        Perform A/B test between two groups.
        
        Args:
            df (pd.DataFrame): Dataset
            group_col (str): Column defining groups
            metric_col (str): Metric to compare
            control_group (str): Control group identifier
            test_group (str): Test group identifier
            
        Returns:
            Dict: A/B test results
        """
        if group_col not in df.columns or metric_col not in df.columns:
            return {"error": f"Required columns not found: {group_col}, {metric_col}"}
        
        control_data = df[df[group_col] == control_group][metric_col].dropna()
        test_data = df[df[group_col] == test_group][metric_col].dropna()
        
        if len(control_data) == 0 or len(test_data) == 0:
            return {"error": "Insufficient data for A/B test"}
        
        results = {
            'test_name': f'A/B Test: {control_group} vs {test_group}',
            'metric': metric_col,
            'control_group': control_group,
            'test_group': test_group,
            'alpha': self.alpha
        }
        
        # Descriptive statistics
        results['control_stats'] = {
            'count': len(control_data),
            'mean': control_data.mean(),
            'std': control_data.std(),
            'median': control_data.median()
        }
        
        results['test_stats'] = {
            'count': len(test_data),
            'mean': test_data.mean(),
            'std': test_data.std(),
            'median': test_data.median()
        }
        
        # Effect size
        pooled_std = np.sqrt(((len(control_data) - 1) * control_data.var() + 
                             (len(test_data) - 1) * test_data.var()) / 
                            (len(control_data) + len(test_data) - 2))
        
        cohens_d = (test_data.mean() - control_data.mean()) / pooled_std
        results['effect_size'] = cohens_d
        
        # Statistical test
        levene_stat, levene_p = levene(control_data, test_data)
        equal_var = levene_p > 0.05
        
        t_stat, p_value = ttest_ind(control_data, test_data, equal_var=equal_var)
        
        results['statistical_test'] = {
            'test_type': f'Independent t-test (equal_var={equal_var})',
            'statistic': t_stat,
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'interpretation': f'Significant difference between {control_group} and {test_group}' if p_value < self.alpha 
                           else f'No significant difference between {control_group} and {test_group}'
        }
        
        # Business interpretation
        lift = ((test_data.mean() - control_data.mean()) / control_data.mean()) * 100
        results['business_metrics'] = {
            'absolute_difference': test_data.mean() - control_data.mean(),
            'relative_lift_percent': lift,
            'practical_significance': 'High' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Low'
        }
        
        return results
    
    def run_all_hypothesis_tests(self, df: pd.DataFrame) -> Dict:
        """
        Run all predefined hypothesis tests for the AlphaCare project.
        
        Args:
            df (pd.DataFrame): Insurance dataset
            
        Returns:
            Dict: All test results
        """
        print("🧪 Running comprehensive hypothesis testing...")
        
        # Calculate risk metrics
        df_with_metrics = self.calculate_risk_metrics(df)
        
        all_results = {
            'dataset_info': {
                'total_records': len(df_with_metrics),
                'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'significance_level': self.alpha
            }
        }
        
        # Test 1: Province risk differences
        print("   Testing province risk differences...")
        province_results = self.test_risk_differences_provinces(df_with_metrics)
        all_results['province_tests'] = province_results
        
        # Test 2: Zip code risk differences
        print("   Testing zip code risk differences...")
        zipcode_results = self.test_risk_differences_zipcodes(df_with_metrics)
        all_results['zipcode_tests'] = zipcode_results
        
        # Test 3: Gender risk differences
        print("   Testing gender risk differences...")
        gender_results = self.test_gender_risk_differences(df_with_metrics)
        all_results['gender_tests'] = gender_results
        
        # Summary of all tests
        test_summary = []
        for test_category, test_data in all_results.items():
            if test_category == 'dataset_info':
                continue
                
            if isinstance(test_data, dict) and 'test_name' in test_data:
                for test_name, test_result in test_data.items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        test_summary.append({
                            'category': test_category,
                            'test': test_name,
                            'p_value': test_result['p_value'],
                            'significant': test_result['reject_null'],
                            'interpretation': test_result['interpretation']
                        })
        
        all_results['test_summary'] = pd.DataFrame(test_summary)
        
        print("✅ All hypothesis tests completed!")
        return all_results
    
    def generate_test_report(self, test_results: Dict) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            test_results (Dict): Results from hypothesis tests
            
        Returns:
            str: Formatted test report
        """
        report = []
        report.append("=" * 80)
        report.append("ALPHACCARE INSURANCE ANALYTICS - HYPOTHESIS TESTING REPORT")
        report.append("=" * 80)
        report.append("")
        
        if 'dataset_info' in test_results:
            info = test_results['dataset_info']
            report.append(f"Dataset: {info['total_records']:,} records")
            report.append(f"Test Date: {info['test_date']}")
            report.append(f"Significance Level: {info['significance_level']}")
            report.append("")
        
        # Summary table
        if 'test_summary' in test_results:
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 40)
            summary_df = test_results['test_summary']
            significant_tests = summary_df[summary_df['significant']].shape[0]
            total_tests = summary_df.shape[0]
            
            report.append(f"Total Tests Performed: {total_tests}")
            report.append(f"Significant Results: {significant_tests}")
            report.append(f"Non-Significant Results: {total_tests - significant_tests}")
            report.append("")
            
            for _, row in summary_df.iterrows():
                status = "✅ SIGNIFICANT" if row['significant'] else "❌ NOT SIGNIFICANT"
                report.append(f"{status}: {row['interpretation']}")
            report.append("")
        
        # Detailed results
        for test_category, test_data in test_results.items():
            if test_category in ['dataset_info', 'test_summary']:
                continue
                
            if isinstance(test_data, dict) and 'test_name' in test_data:
                report.append(f"DETAILED RESULTS: {test_data['test_name'].upper()}")
                report.append("-" * 60)
                report.append(f"Null Hypothesis: {test_data.get('null_hypothesis', 'Not specified')}")
                report.append("")
                
                for test_name, test_result in test_data.items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        report.append(f"  {test_name.replace('_', ' ').title()}:")
                        report.append(f"    Test Type: {test_result['test_type']}")
                        report.append(f"    Test Statistic: {test_result['statistic']:.4f}")
                        report.append(f"    P-value: {test_result['p_value']:.6f}")
                        report.append(f"    Result: {test_result['interpretation']}")
                        report.append("")
                
                report.append("")
        
        return "\n".join(report)


# Utility functions
def quick_hypothesis_testing(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Perform quick hypothesis testing on a dataset.
    
    Args:
        df (pd.DataFrame): Insurance dataset
        alpha (float): Significance level
        
    Returns:
        Dict: Test results
    """
    tester = InsuranceStatsTester(alpha=alpha)
    return tester.run_all_hypothesis_tests(df)


if __name__ == "__main__":
    # Example usage with sample data
    from data_processing import load_sample_data
    
    sample_df = load_sample_data()
    results = quick_hypothesis_testing(sample_df)
    
    tester = InsuranceStatsTester()
    report = tester.generate_test_report(results)
    print(report)
