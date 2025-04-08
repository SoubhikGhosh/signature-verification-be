from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import os
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor
from contextlib import contextmanager
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("signature-similarity-api")

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD2ArK74wBtL1ufYmpyrV2LqaOBrSi3mlU")

if not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY environment variable")
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv("DB_NAME", "cheque_ocr"),
    'user': os.getenv("DB_USER", "soubhikghosh"),
    'password': os.getenv("DB_PASSWORD", "99Ghosh"),
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

# Configure the Gemini API
logger.info("Configuring Gemini API")
genai.configure(api_key=GOOGLE_API_KEY)

# Create FastAPI app
app = FastAPI(title="Signature Similarity API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SimilarityResponse(BaseModel):
    similarity_score: float
    analysis: str
    comparison_id: Optional[str] = None

class DatabaseStatus(BaseModel):
    connected: bool
    tables_exist: bool
    error: Optional[str] = None

# Database connection management
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        logger.info("Establishing database connection")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")

@contextmanager
def get_db_cursor(conn=None, cursor_factory=DictCursor):
    """Context manager for database cursors"""
    if conn is None:
        # Create a new connection if one is not provided
        with get_db_connection() as new_conn:
            with new_conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur
            new_conn.commit()
    else:
        # Use the provided connection
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur

def init_db():
    """Initialize database tables if they don't exist"""
    logger.info("Initializing database tables")
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Create signature_comparisons table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS signature_comparisons (
                    id UUID PRIMARY KEY,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    signature1_filename TEXT,
                    signature2_filename TEXT,
                    signature1_content_type TEXT,
                    signature2_content_type TEXT,
                    similarity_score FLOAT,
                    analysis TEXT,
                    request_time FLOAT,
                    response_data JSONB
                );
                """)
                
                # Create signature_images table for storing the actual images
                cur.execute("""
                CREATE TABLE IF NOT EXISTS signature_images (
                    id UUID PRIMARY KEY,
                    comparison_id UUID REFERENCES signature_comparisons(id) ON DELETE CASCADE,
                    signature_number INTEGER CHECK (signature_number IN (1, 2)),
                    image_data BYTEA,
                    UNIQUE (comparison_id, signature_number)
                );
                """)
                
                conn.commit()
                logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Generate a unique request ID
    request_id = os.urandom(8).hex()
    
    # Log the request
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    # Process the request
    response = await call_next(request)
    
    # Calculate and log processing time
    process_time = time.time() - start_time
    logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
    
    return response

# Database operations
async def save_comparison_to_db(
    background_tasks: BackgroundTasks,
    comparison_id: str,
    signature1: UploadFile,
    signature2: UploadFile,
    sig1_content: bytes,
    sig2_content: bytes,
    similarity_score: float,
    analysis: str,
    request_time: float,
    response_data: dict
):
    """Save comparison data to database in the background"""
    
    def _save_to_db():
        try:
            logger.info(f"Saving comparison {comparison_id} to database")
            with get_db_connection() as conn:
                with get_db_cursor(conn) as cur:
                    # Insert into signature_comparisons
                    cur.execute("""
                    INSERT INTO signature_comparisons (
                        id, signature1_filename, signature2_filename, 
                        signature1_content_type, signature2_content_type,
                        similarity_score, analysis, request_time, response_data
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        comparison_id,
                        signature1.filename,
                        signature2.filename,
                        signature1.content_type,
                        signature2.content_type,
                        similarity_score,
                        analysis,
                        request_time,
                        json.dumps(response_data)
                    ))
                    
                    # Insert signature 1 image
                    cur.execute("""
                    INSERT INTO signature_images (id, comparison_id, signature_number, image_data)
                    VALUES (%s, %s, %s, %s)
                    """, (
                        str(uuid.uuid4()),
                        comparison_id,
                        1,
                        sig1_content
                    ))
                    
                    # Insert signature 2 image
                    cur.execute("""
                    INSERT INTO signature_images (id, comparison_id, signature_number, image_data)
                    VALUES (%s, %s, %s, %s)
                    """, (
                        str(uuid.uuid4()),
                        comparison_id,
                        2,
                        sig2_content
                    ))
                    
                conn.commit()
                logger.info(f"Comparison {comparison_id} saved to database successfully")
        except Exception as e:
            logger.error(f"Error saving comparison to database: {e}", exc_info=True)
    
    # Add to background tasks to avoid slowing down the API response
    background_tasks.add_task(_save_to_db)

# API endpoints
@app.post("/compare-signatures/", response_model=SimilarityResponse)
async def compare_signatures(
    background_tasks: BackgroundTasks,
    signature1: UploadFile = File(...),
    signature2: UploadFile = File(...),
):
    """
    Compare two signature images and return a similarity score.
    Also stores the comparison data in the database.
    """
    logger.info(f"Received signature comparison request: {signature1.filename} vs {signature2.filename}")
    start_request_time = time.time()
    comparison_id = str(uuid.uuid4())
    logger.info(f"Generated comparison ID: {comparison_id}")
    
    try:
        # Read the uploaded files
        sig1_content = await signature1.read()
        sig2_content = await signature2.read()
        
        logger.info(f"Signature 1 size: {len(sig1_content)} bytes, type: {signature1.content_type}")
        logger.info(f"Signature 2 size: {len(sig2_content)} bytes, type: {signature2.content_type}")

        # Check if files are images
        if not signature1.content_type.startswith("image/") or not signature2.content_type.startswith("image/"):
            logger.warning(f"Invalid file types: {signature1.content_type}, {signature2.content_type}")
            raise HTTPException(status_code=400, detail="Uploaded files must be images")

        # Encode images to base64 for Gemini
        sig1_b64 = base64.b64encode(sig1_content).decode("utf-8")
        sig2_b64 = base64.b64encode(sig2_content).decode("utf-8")

        # Create Gemini model
        logger.info("Initializing Gemini 1.5 Pro model")
        model = genai.GenerativeModel('gemini-1.5-pro')

        # Define the prompt for signature comparison
        prompt = """
        You are a forensic handwriting expert analyzing two signature images. 
        Compare these two signatures and determine their similarity.
        
        Focus on:
        1. Overall shape and flow
        2. Pressure points and line quality
        3. Specific features unique to the signer
        4. Start and end points
        5. Proportions and spacing
        
        Provide a similarity score from 0.0 to 1.0 where:
        - 0.0 means completely different signatures
        - 1.0 means identical signatures
        
        Return your response in this format:
        Score: [similarity score as a decimal]
        Analysis: [brief explanation of your reasoning]
        """

        # Prepare the content parts with the images
        parts = [
            {
                "text": prompt
            },
            {
                "inline_data": {
                    "mime_type": signature1.content_type,
                    "data": sig1_b64
                }
            },
            {
                "inline_data": {
                    "mime_type": signature2.content_type,
                    "data": sig2_b64
                }
            }
        ]

        # Generate response from Gemini
        logger.info("Sending request to Gemini for signature analysis")
        start_time = time.time()
        response = model.generate_content(parts)
        process_time = time.time() - start_time
        logger.info(f"Received Gemini response in {process_time:.2f}s")
        
        # Extract the score and analysis from the response
        text_response = response.text
        logger.debug(f"Raw Gemini response: {text_response[:100]}...")
        
        # Parse the response to extract score and analysis
        lines = text_response.strip().split('\n')
        
        score_line = next((line for line in lines if line.startswith('Score:')), None)
        if not score_line:
            logger.error("Could not find similarity score in Gemini response")
            raise ValueError("Could not find similarity score in Gemini response")
            
        try:
            similarity_score = float(score_line.split(':')[1].strip())
            # Ensure score is within valid range
            similarity_score = max(0.0, min(1.0, similarity_score))
            logger.info(f"Similarity score calculated: {similarity_score}")
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid similarity score format: {e}")
            raise ValueError("Invalid similarity score format in Gemini response")
        
        # Extract analysis (everything after "Analysis:")
        analysis_start_idx = text_response.find("Analysis:")
        if analysis_start_idx == -1:
            analysis = "No detailed analysis provided."
        else:
            analysis = text_response[analysis_start_idx + len("Analysis:"):].strip()
        
        # Get the processed response
        response_data = {
            "similarity_score": similarity_score,
            "analysis": analysis,
            "comparison_id": comparison_id
        }
        
        # Calculate total request time
        total_request_time = time.time() - start_request_time
        logger.info(f"Total request processing time: {total_request_time:.2f}s")
        
        # Save to database in the background
        await save_comparison_to_db(
            background_tasks,
            comparison_id,
            signature1,
            signature2,
            sig1_content,
            sig2_content,
            similarity_score,
            analysis,
            total_request_time,
            response_data
        )
        
        logger.info("Successfully completed signature comparison")
        return SimilarityResponse(
            similarity_score=similarity_score,
            analysis=analysis,
            comparison_id=comparison_id
        )
            
    except Exception as e:
        logger.error(f"Error processing signatures: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing signatures: {str(e)}")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Signature Similarity API is running. Use /compare-signatures/ endpoint to compare signature images."}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is functioning properly.
    Returns service status, version, and uptime information.
    """
    logger.info("Health check endpoint accessed")
    
    # Check if Gemini API is configured
    api_status = "healthy" if GOOGLE_API_KEY else "unhealthy"
    
    # Check database connection
    db_status = "unhealthy"
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("SELECT 1")
                if cur.fetchone():
                    db_status = "healthy"
    except Exception as e:
        logger.warning(f"Database connection failed during health check: {e}")
    
    # Get system information
    uptime_seconds = time.time() - startup_time
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    health_data = {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s",
        "gemini_api": api_status,
        "database": db_status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }
    
    # If any component is unhealthy, the overall status is unhealthy
    if api_status == "unhealthy" or db_status == "unhealthy":
        health_data["status"] = "unhealthy"
    
    return health_data

@app.get("/db-status", response_model=DatabaseStatus)
async def check_db_status():
    """
    Check database connection and table status.
    Useful for monitoring and troubleshooting.
    """
    logger.info("Database status check requested")
    result = DatabaseStatus(connected=False, tables_exist=False)
    
    try:
        with get_db_connection() as conn:
            result.connected = True
            
            with get_db_cursor(conn) as cur:
                # Check if our tables exist
                cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'signature_comparisons'
                ) AND EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'signature_images'
                )
                """)
                result.tables_exist = cur.fetchone()[0]
                
        logger.info(f"Database status: connected={result.connected}, tables_exist={result.tables_exist}")
        return result
    except Exception as e:
        logger.error(f"Database status check failed: {e}", exc_info=True)
        result.error = str(e)
        return result

@app.get("/comparisons/{comparison_id}")
async def get_comparison(comparison_id: str):
    """
    Retrieve a stored comparison by ID.
    """
    logger.info(f"Retrieving comparison with ID: {comparison_id}")
    
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                cur.execute("""
                SELECT id, created_at, signature1_filename, signature2_filename, 
                       similarity_score, analysis, request_time
                FROM signature_comparisons
                WHERE id = %s
                """, (comparison_id,))
                
                result = cur.fetchone()
                
                if not result:
                    logger.warning(f"Comparison not found: {comparison_id}")
                    raise HTTPException(status_code=404, detail="Comparison not found")
                
                # Convert to dictionary
                comparison = dict(result)
                # Convert datetime to string for JSON serialization
                comparison['created_at'] = comparison['created_at'].isoformat()
                
                logger.info(f"Successfully retrieved comparison {comparison_id}")
                return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/comparisons")
async def list_comparisons(limit: int = 10, offset: int = 0):
    """
    List stored comparisons with pagination.
    """
    logger.info(f"Listing comparisons with limit={limit}, offset={offset}")
    
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Get total count
                cur.execute("SELECT COUNT(*) FROM signature_comparisons")
                total = cur.fetchone()[0]
                
                # Get paginated results
                cur.execute("""
                SELECT id, created_at, signature1_filename, signature2_filename, 
                       similarity_score, analysis
                FROM signature_comparisons
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """, (limit, offset))
                
                results = cur.fetchall()
                
                # Convert to list of dictionaries
                comparisons = []
                for row in results:
                    comparison = dict(row)
                    comparison['created_at'] = comparison['created_at'].isoformat()
                    comparisons.append(comparison)
                
                logger.info(f"Retrieved {len(comparisons)} comparisons")
                return {
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "comparisons": comparisons
                }
    except Exception as e:
        logger.error(f"Error listing comparisons: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/comparisons/{comparison_id}")
async def delete_comparison(comparison_id: str):
    """
    Delete a stored comparison by ID.
    """
    logger.info(f"Deleting comparison with ID: {comparison_id}")
    
    try:
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cur:
                # Check if the comparison exists
                cur.execute("SELECT 1 FROM signature_comparisons WHERE id = %s", (comparison_id,))
                if cur.fetchone() is None:
                    logger.warning(f"Comparison not found for deletion: {comparison_id}")
                    raise HTTPException(status_code=404, detail="Comparison not found")
                
                # Delete the comparison (cascade will delete related images)
                cur.execute("DELETE FROM signature_comparisons WHERE id = %s", (comparison_id,))
                conn.commit()
                
                logger.info(f"Successfully deleted comparison {comparison_id}")
                return {"message": "Comparison deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def initialize_database():
    """
    Initialize or reset the database.
    """
    logger.info("Database initialization requested via API")
    
    try:
        init_db()
        return {"status": "success", "message": "Database initialized successfully"}
    except Exception as e:
        logger.error(f"API-requested database initialization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

# Track application startup time
startup_time = time.time()

# Initialize database on startup
try:
    init_db()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    # Continue running the application even if DB init fails

logger.info("Application initialized")

if __name__ == "__main__":
    logger.info("Starting Signature Similarity API server")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)