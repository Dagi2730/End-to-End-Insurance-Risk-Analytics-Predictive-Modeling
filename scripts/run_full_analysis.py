"""
Complete analysis pipeline runner.
Executes all notebooks in sequence and generates comprehensive reports.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
import pandas as pd
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AnalysisPipelineRunner:
    """Orchestrates the complete insurance analytics pipeline."""
    
    def __init__(self, data_path: str, output_dir: str = "results"):
        """
        Initialize the pipeline runner.
        
        Args:
            data_path: Path to the insurance dataset
            output_dir: Directory to save results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.notebooks_dir = "notebooks"
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Pipeline initialized with data: {data_path}")
    
    def validate_environment(self):
        """Validate that all required dependencies are installed."""
        logger.info("Validating environment...")
        
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'xgboost', 
            'matplotlib', 'seaborn', 'plotly', 'streamlit',
            'shap', 'statsmodels', 'scipy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            raise ImportError(f"Please install missing packages: {missing_packages}")
        
        logger.info("Environment validation successful")
    
    def validate_data(self):
        """Validate the input data file."""
        logger.info("Validating input data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {df.shape}")
            
            # Check for required columns
            required_columns = [
                'TotalPremium', 'TotalClaims', 'Province', 'Gender',
                'VehicleType', 'SumInsured', 'CalculatedPremiumPerTerm'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing recommended columns: {missing_columns}")
            
            self.results['data_validation'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'missing_columns': missing_columns
            }
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
        
        logger.info("Data validation successful")
    
    def run_notebook(self, notebook_name: str):
        """Execute a Jupyter notebook and capture results."""
        notebook_path = os.path.join(self.notebooks_dir, notebook_name)
        
        if not os.path.exists(notebook_path):
            logger.error(f"Notebook not found: {notebook_path}")
            return False
        
        logger.info(f"Executing notebook: {notebook_name}")
        
        try:
            # Execute notebook using nbconvert
            cmd = [
                'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                '--inplace', notebook_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info(f"Successfully executed: {notebook_name}")
                return True
            else:
                logger.error(f"Notebook execution failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Notebook execution timed out: {notebook_name}")
            return False
        except Exception as e:
            logger.error(f"Error executing notebook {notebook_name}: {e}")
            return False
    
    def run_eda_analysis(self):
        """Run exploratory data analysis."""
        logger.info("Starting EDA analysis...")
        success = self.run_notebook("1_EDA.ipynb")
        self.results['eda_completed'] = success
        return success
    
    def run_hypothesis_testing(self):
        """Run statistical hypothesis testing."""
        logger.info("Starting hypothesis testing...")
        success = self.run_notebook("2_Hypothesis_Testing.ipynb")
        self.results['hypothesis_testing_completed'] = success
        return success
    
    def run_predictive_modeling(self):
        """Run predictive modeling pipeline."""
        logger.info("Starting predictive modeling...")
        success = self.run_notebook("3_Predictive_Modeling.ipynb")
        self.results['modeling_completed'] = success
        return success
    
    def run_tests(self):
        """Run the test suite."""
        logger.info("Running test suite...")
        
        try:
            # Run pytest
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/', '-v'],
                capture_output=True, text=True
            )
            
            test_success = result.returncode == 0
            self.results['tests_passed'] = test_success
            
            if test_success:
                logger.info("All tests passed")
            else:
                logger.warning(f"Some tests failed: {result.stdout}")
            
            return test_success
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            self.results['tests_passed'] = False
            return False
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        logger.info("Generating summary report...")
        
        report = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'data_path': self.data_path,
                'results': self.results
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, 'pipeline_summary.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved: {report_path}")
        return report
    
    def _generate_recommendations(self):
        """Generate business recommendations based on results."""
        recommendations = []
        
        if self.results.get('eda_completed'):
            recommendations.append(
                "EDA completed successfully - review visualizations for business insights"
            )
        
        if self.results.get('hypothesis_testing_completed'):
            recommendations.append(
                "Statistical tests completed - implement risk-based pricing strategies"
            )
        
        if self.results.get('modeling_completed'):
            recommendations.append(
                "Predictive models trained - deploy best performing model for premium prediction"
            )
        
        if self.results.get('tests_passed'):
            recommendations.append(
                "All tests passed - code is ready for production deployment"
            )
        
        return recommendations
    
    def run_complete_pipeline(self):
        """Execute the complete analysis pipeline."""
        logger.info("Starting complete analysis pipeline...")
        
        try:
            # Validation steps
            self.validate_environment()
            self.validate_data()
            
            # Analysis steps
            eda_success = self.run_eda_analysis()
            hypothesis_success = self.run_hypothesis_testing()
            modeling_success = self.run_predictive_modeling()
            
            # Testing
            tests_success = self.run_tests()
            
            # Generate report
            report = self.generate_summary_report()
            
            # Overall success
            overall_success = all([
                eda_success, hypothesis_success, modeling_success, tests_success
            ])
            
            if overall_success:
                logger.info("Pipeline completed successfully!")
            else:
                logger.warning("Pipeline completed with some issues")
            
            return overall_success, report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False, None


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AlphaCare Insurance Analytics Pipeline')
    parser.add_argument('--data', required=True, help='Path to insurance dataset CSV file')
    parser.add_argument('--output', default='results', help='Output directory for results')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    runner = AnalysisPipelineRunner(args.data, args.output)
    
    if args.skip_tests:
        runner.results['tests_passed'] = True  # Skip tests
    
    success, report = runner.run_complete_pipeline()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Results saved in: {args.output}")
        print("🚀 Ready to launch Streamlit dashboard!")
    else:
        print("\n❌ Pipeline completed with issues")
        print("📋 Check logs for details")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
