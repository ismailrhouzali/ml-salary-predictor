"""
Developer Salary Prediction - Application Launcher
Run this file to start the Streamlit application
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    
    # Set the path to the app file
    app_path = os.path.join(os.path.dirname(__file__), 'src', 'app.py')
    
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())
