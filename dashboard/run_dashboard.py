import subprocess
import sys
import os

def run_streamlit():
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dashboard_dir)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ], check=True)

if __name__ == "__main__":
    print("🚀 Launching AlphaCare Insurance Dashboard...")
    run_streamlit()
