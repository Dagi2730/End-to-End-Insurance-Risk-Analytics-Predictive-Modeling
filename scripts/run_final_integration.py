#!/usr/bin/env python3
"""
Execute Final Integration and Performance Benchmarking
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Execute final integration and benchmarking."""
    print("🎉 ALPHACARE INSURANCE ANALYTICS - FINAL EXECUTION")
    print("=" * 70)
    print("Running comprehensive system validation and performance benchmarking...")
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run final integration validation
    integration_success = run_script(
        'scripts/final_integration.py',
        'Final System Integration & Validation'
    )
    
    # Run performance benchmarking
    benchmark_success = run_script(
        'scripts/performance_benchmark.py', 
        'Comprehensive Performance Benchmarking'
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 FINAL EXECUTION SUMMARY")
    print("=" * 70)
    
    if integration_success and benchmark_success:
        print("🎉 SUCCESS! AlphaCare Insurance Analytics Platform is COMPLETE!")
        print("\n✅ System Integration: PASSED")
        print("✅ Performance Benchmarking: COMPLETED")
        print("✅ All Components Validated: READY FOR PRODUCTION")
        
        print("\n🚀 PLATFORM CAPABILITIES:")
        print("   • Advanced Data Processing & EDA")
        print("   • Statistical Hypothesis Testing")
        print("   • Machine Learning Models (Linear, RF, XGBoost)")
        print("   • Neural Networks & AutoML")
        print("   • Interactive Streamlit Dashboard")
        print("   • Real-time WebSocket Dashboard")
        print("   • REST API with FastAPI")
        print("   • Automated Reporting & BI")
        print("   • Real-time Monitoring & Alerting")
        print("   • Enterprise Security & Performance Optimization")
        print("   • Docker Deployment & CI/CD Pipeline")
        
        print("\n📊 RESULTS AVAILABLE:")
        print("   • results/final_validation_report.json")
        print("   • results/performance_benchmark.json")
        print("   • logs/final_integration.log")
        
        print("\n🎯 NEXT STEPS:")
        print("   1. Review validation and benchmark reports")
        print("   2. Deploy using: docker-compose up")
        print("   3. Access dashboard at: http://localhost:8501")
        print("   4. Access API at: http://localhost:8000")
        
        return True
    else:
        print("❌ EXECUTION INCOMPLETE")
        if not integration_success:
            print("   • System Integration: FAILED")
        if not benchmark_success:
            print("   • Performance Benchmarking: FAILED")
        print("\n🔧 Please check logs for details and resolve issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
