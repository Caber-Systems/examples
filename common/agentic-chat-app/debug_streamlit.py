import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    # This is the path to your streamlit app
    app_path = "/CaberApp/agentic-app/app.py"
    
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        # You can add streamlit command line arguments here
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    
    stcli.main()
