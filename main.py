"""
Main entry point for the University Matching Algorithm API.
"""
import uvicorn
import logging
from src.api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting University Matching Algorithm API...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Set to True for development
        log_level="info"
    ) 