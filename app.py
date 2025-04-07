from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import os
import logging
import io
import json
import psycopg2
from psycopg2.extras import Json, DictCursor
from typing import Dict, Any, List, Optional
from datetime import datetime
from PIL import Image
import uvicorn
import hashlib
from pydantic import BaseModel
import numpy as np
from io import BytesIO
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Signature Verification API",
    description="API for verifying signatures using Google Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google API - replace with your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD2ArK74wBtL1ufYmpyrV2LqaOBrSi3mlU")
genai.configure(api_key=GOOGLE_API_KEY)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# Database configuration
DB_CONFIG = {
    'dbname': 'signature_verification',
    'user': os.getenv("DB_USER", "soubhikghosh"),
    'password': os.getenv("DB_PASSWORD", "99Ghosh"),
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

def get_db_connection():
    """Create the database if it doesn't exist and return a connection."""
    try:
        # First try to connect to the default postgres database to check if our database exists
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if our database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
        exists = cursor.fetchone()
        
        # Create database if it doesn't exist
        if not exists:
            logger.info(f"Database '{DB_CONFIG['dbname']}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            logger.info(f"Database '{DB_CONFIG['dbname']}' created successfully")
        
        cursor.close()
        conn.close()
        
        # Now connect to our actual database
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"Connected to database '{DB_CONFIG['dbname']}'")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def init_db():
    """Initialize database tables if they don't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create reference_signatures table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reference_signatures (
            id SERIAL PRIMARY KEY,
            person_id TEXT NOT NULL,
            person_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(person_id, file_hash)
        )
        ''')
        
        # Create verification_results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_results (
            id SERIAL PRIMARY KEY,
            verification_id TEXT NOT NULL UNIQUE,
            person_id TEXT NOT NULL,
            reference_signature_id INTEGER REFERENCES reference_signatures(id),
            sample_signature_filename TEXT NOT NULL,
            sample_signature_path TEXT NOT NULL,
            is_match BOOLEAN NOT NULL,
            confidence FLOAT NOT NULL, 
            analysis TEXT NOT NULL,
            verification_time TIMESTAMP DEFAULT NOW()
        )
        ''')
        
        # Create verification_features table for detailed feature comparison
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_features (
            id SERIAL PRIMARY KEY,
            verification_id TEXT NOT NULL REFERENCES verification_results(verification_id),
            feature_name TEXT NOT NULL,
            match_score FLOAT NOT NULL,
            description TEXT NOT NULL
        )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_reference_signatures_person_id ON reference_signatures(person_id);
        CREATE INDEX IF NOT EXISTS idx_verification_results_person_id ON verification_results(person_id);
        CREATE INDEX IF NOT EXISTS idx_verification_features_verification_id ON verification_features(verification_id);
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

class SignatureProcessor:
    """Helper class for signature verification operations using Gemini"""
    
    @staticmethod
    def save_upload_file(upload_file: UploadFile, directory: str = "uploads") -> str:
        """Save an uploaded file and return the file path."""
        try:
            # Create a unique filename
            file_extension = os.path.splitext(upload_file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(directory, unique_filename)
            
            # Reset file pointer and read content
            upload_file.file.seek(0)
            content = upload_file.file.read()
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(content)
                
            return file_path, unique_filename, content
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
    
    @staticmethod
    def register_reference_signature(person_id: str, person_name: str, file: UploadFile) -> Dict[str, Any]:
        """Register a reference signature for a person."""
        try:
            # Save uploaded file
            file_path, filename, file_content = SignatureProcessor.save_upload_file(file)
            
            # Calculate file hash
            file_hash = hashlib.md5(file_content).hexdigest()
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if this exact signature already exists for this person
            cursor.execute(
                "SELECT id FROM reference_signatures WHERE person_id = %s AND file_hash = %s",
                (person_id, file_hash)
            )
            existing = cursor.fetchone()
            
            if existing:
                cursor.close()
                conn.close()
                return {
                    "status": "already_exists",
                    "message": "This exact signature is already registered for this person",
                    "signature_id": existing[0]
                }
            
            # Insert new reference signature
            cursor.execute(
                """
                INSERT INTO reference_signatures 
                (person_id, person_name, filename, file_path, file_hash)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (person_id, person_name, filename, file_path, file_hash)
            )
            
            signature_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            return {
                "status": "success",
                "message": "Reference signature registered successfully",
                "signature_id": signature_id,
                "person_id": person_id,
                "person_name": person_name,
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"Error registering reference signature: {str(e)}")
            raise
    
    @staticmethod
    def verify_signature(person_id: str, sample_file: UploadFile) -> Dict[str, Any]:
        """Verify a signature against the reference signatures for a person."""
        try:
            # Save uploaded sample signature
            sample_file_path, sample_filename, sample_content = SignatureProcessor.save_upload_file(sample_file)
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get reference signatures for this person
            cursor.execute(
                "SELECT id, file_path FROM reference_signatures WHERE person_id = %s ORDER BY created_at DESC LIMIT 1",
                (person_id,)
            )
            reference = cursor.fetchone()
            
            if not reference:
                cursor.close()
                conn.close()
                raise HTTPException(
                    status_code=404,
                    detail="No reference signature found for this person"
                )
            
            reference_id, reference_path = reference
            
            # Generate a unique verification ID
            verification_id = str(uuid.uuid4())
            
            # Perform verification using Gemini
            verification_result = SignatureProcessor._analyze_signatures(reference_path, sample_file_path)
            
            # Save verification result
            cursor.execute(
                """
                INSERT INTO verification_results
                (verification_id, person_id, reference_signature_id, sample_signature_filename, 
                sample_signature_path, is_match, confidence, analysis)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    verification_id,
                    person_id,
                    reference_id,
                    sample_filename,
                    sample_file_path,
                    verification_result["is_match"],
                    verification_result["confidence"],
                    json.dumps(verification_result["analysis"])
                )
            )
            
            # Save detailed feature comparisons
            for feature in verification_result.get("features", []):
                cursor.execute(
                    """
                    INSERT INTO verification_features
                    (verification_id, feature_name, match_score, description)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        verification_id,
                        feature["name"],
                        feature["score"],
                        feature["description"]
                    )
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return {
                "verification_id": verification_id,
                "person_id": person_id,
                "is_match": verification_result["is_match"],
                "confidence": verification_result["confidence"],
                "analysis": verification_result["analysis"],
                "features": verification_result.get("features", []),
                "verification_time": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error verifying signature: {str(e)}")
            raise
    
    @staticmethod
    def _analyze_signatures(reference_path: str, sample_path: str) -> Dict[str, Any]:
        """Analyze and compare two signatures using Gemini."""
        try:
            # Load images
            reference_img = Image.open(reference_path)
            sample_img = Image.open(sample_path)
            
            # Convert to bytes for Gemini
            reference_buffer = BytesIO()
            sample_buffer = BytesIO()
            
            reference_img.save(reference_buffer, format="PNG")
            sample_img.save(sample_buffer, format="PNG")
            
            reference_bytes = reference_buffer.getvalue()
            sample_bytes = sample_buffer.getvalue()
            
            return SignatureProcessor._compare_signature_bytes(reference_bytes, sample_bytes)
        except Exception as e:
            logger.error(f"Error analyzing signatures: {str(e)}")
            return {
                "is_match": False,
                "confidence": 0.0,
                "analysis": f"Error during analysis: {str(e)}",
                "features": []
            }
    
    @staticmethod
    def _compare_signature_bytes(reference_bytes: bytes, sample_bytes: bytes) -> Dict[str, Any]:
            
            # Initialize Gemini model with multimodal capabilities
            model = genai.GenerativeModel("gemini-1.5-pro")
            
            # Prompt for signature verification
            verification_prompt = """
            You are a professional forensic signature verification expert. Analyze these two signatures:
            
            1. The first image is the REFERENCE signature (known to be authentic)
            2. The second image is the SAMPLE signature that needs verification
            
            Compare them with expert precision by analyzing the following features:
            1. Overall shape and form
            2. Line quality and pen pressure
            3. Writing speed and rhythm 
            4. Starting and ending strokes
            5. Proportions and spacing
            6. Connections between characters/elements
            7. Slant and slope consistency
            8. Complexity and distinctive elements
            
            For each feature, assign a match score from 0.0 to 1.0 where:
            - 0.0 = No match, completely different
            - 0.5 = Some similarities but significant differences
            - 1.0 = Perfect match
            
            Then provide:
            1. An overall confidence score (0.0-1.0) for whether these signatures match
            2. A determination of "match" or "different" based on your analysis
            3. A brief explanation of your conclusion
            
            Return ONLY a JSON object with the following structure:
            {
                "is_match": true/false,
                "confidence": 0.0-1.0,
                "analysis": "detailed explanation of findings",
                "features": [
                    {
                        "name": "feature name",
                        "score": 0.0-1.0,
                        "description": "analysis of this feature"
                    },
                    ...
                ]
            }
            """
            
            # Make API call to Gemini
            response = model.generate_content([
                verification_prompt,
                {"mime_type": "image/png", "data": reference_bytes},
                {"mime_type": "image/png", "data": sample_bytes}
            ])
            
            # Parse and return results
            result = SignatureProcessor._extract_json(response.text)
            
            # Ensure we have required fields
            if "is_match" not in result or "confidence" not in result:
                result = {
                    "is_match": False,
                    "confidence": 0.0,
                    "analysis": "Failed to analyze signatures properly",
                    "features": []
                }
            
            return result
            
    @staticmethod
    def compare_two_signatures(signature1_file: UploadFile, signature2_file: UploadFile) -> Dict[str, Any]:
        """Compare two uploaded signature images directly."""
        try:
            # Save uploaded signatures temporarily
            sig1_path, sig1_filename, sig1_content = SignatureProcessor.save_upload_file(signature1_file)
            sig2_path, sig2_filename, sig2_content = SignatureProcessor.save_upload_file(signature2_file)
            
            # Analyze signatures
            comparison_result = SignatureProcessor._analyze_signatures(sig1_path, sig2_path)
            
            # Add file information
            comparison_result["signature1_filename"] = sig1_filename
            comparison_result["signature2_filename"] = sig2_filename
            comparison_result["comparison_time"] = datetime.now().isoformat()
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error comparing signatures: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Error analyzing signatures: {str(e)}")
            return {
                "is_match": False,
                "confidence": 0.0,
                "analysis": f"Error during analysis: {str(e)}",
                "features": []
            }
    
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text that might contain markdown or extra content."""
        import json
        import re
        
        # Clean up potential JSON formatting
        if '```json' in text:
            # Extract content between ```json and ```
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
        elif '```' in text:
            # Extract content between ``` and ```
            match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
                
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}, Text: {text[:100]}...")
            return {}
    
    @staticmethod
    def get_verification_history(person_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get verification history for a person."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            cursor.execute(
                """
                SELECT 
                    vr.verification_id, 
                    vr.is_match, 
                    vr.confidence, 
                    vr.sample_signature_filename,
                    vr.verification_time,
                    rs.person_name
                FROM verification_results vr
                JOIN reference_signatures rs ON vr.reference_signature_id = rs.id
                WHERE vr.person_id = %s
                ORDER BY vr.verification_time DESC
                LIMIT %s
                """,
                (person_id, limit)
            )
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['verification_time'] = result['verification_time'].isoformat()
                results.append(result)
            
            cursor.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting verification history: {str(e)}")
            raise

class PersonBase(BaseModel):
    person_id: str
    person_name: str

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# Routes
@app.post("/register-signature/")
async def register_signature(
    person_id: str = Form(...),
    person_name: str = Form(...),
    signature: UploadFile = File(...)
):
    """
    Register a reference signature for a person.
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(signature.filename)[1].lower()
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload a JPG or PNG image."
            )
        
        result = SignatureProcessor.register_reference_signature(person_id, person_name, signature)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering signature: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-signature/")
async def verify_signature(
    person_id: str = Form(...),
    signature_sample: UploadFile = File(...)
):
    """
    Verify a signature sample against the reference signatures for a person.
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(signature_sample.filename)[1].lower()
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload a JPG or PNG image."
            )
        
        result = SignatureProcessor.verify_signature(person_id, signature_sample)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying signature: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/verification-history/{person_id}")
async def get_verification_history(person_id: str, limit: int = 10):
    """
    Get verification history for a person.
    """
    try:
        results = SignatureProcessor.get_verification_history(person_id, limit)
        return JSONResponse(content={"history": results})
    
    except Exception as e:
        logger.error(f"Error getting verification history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/verification/{verification_id}")
async def get_verification_details(verification_id: str):
    """
    Get detailed information about a specific verification.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get verification details
        cursor.execute(
            """
            SELECT 
                vr.verification_id, 
                vr.person_id,
                vr.is_match, 
                vr.confidence, 
                vr.analysis,
                vr.sample_signature_path,
                vr.verification_time,
                rs.person_name,
                rs.file_path as reference_signature_path
            FROM verification_results vr
            JOIN reference_signatures rs ON vr.reference_signature_id = rs.id
            WHERE vr.verification_id = %s
            """,
            (verification_id,)
        )
        
        verification = cursor.fetchone()
        
        if not verification:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Verification not found")
        
        # Convert to dict and format date
        verification_dict = dict(verification)
        verification_dict['verification_time'] = verification_dict['verification_time'].isoformat()
        
        # Parse JSON analysis field
        try:
            verification_dict['analysis'] = json.loads(verification_dict['analysis'])
        except:
            pass  # Keep as string if not valid JSON
        
        # Get feature details
        cursor.execute(
            """
            SELECT feature_name, match_score, description
            FROM verification_features
            WHERE verification_id = %s
            """,
            (verification_id,)
        )
        
        features = []
        for row in cursor.fetchall():
            features.append({
                "name": row["feature_name"],
                "score": row["match_score"],
                "description": row["description"]
            })
        
        verification_dict['features'] = features
        
        cursor.close()
        conn.close()
        
        return JSONResponse(content=verification_dict)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting verification details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signature-image/{image_type}/{verification_id}")
async def get_signature_image(image_type: str, verification_id: str):
    """
    Get the reference or sample signature image for a verification.
    
    image_type: 'reference' or 'sample'
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if image_type == 'reference':
            cursor.execute(
                """
                SELECT rs.file_path
                FROM verification_results vr
                JOIN reference_signatures rs ON vr.reference_signature_id = rs.id
                WHERE vr.verification_id = %s
                """,
                (verification_id,)
            )
        elif image_type == 'sample':
            cursor.execute(
                """
                SELECT sample_signature_path
                FROM verification_results
                WHERE verification_id = %s
                """,
                (verification_id,)
            )
        else:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid image type. Use 'reference' or 'sample'")
        
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Image not found")
        
        file_path = result[0]
        cursor.close()
        conn.close()
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Return image
        with open(file_path, "rb") as f:
            image_data = f.read()
        
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/png"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving signature image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-signatures/")
async def compare_signatures(
    signature1: UploadFile = File(...),
    signature2: UploadFile = File(...)
):
    """
    Compare two signature images directly without storing them in a database.
    
    This endpoint allows comparing any two signatures without requiring registration.
    """
    try:
        # Validate file types
        for sig in [signature1, signature2]:
            file_extension = os.path.splitext(sig.filename)[1].lower()
            if file_extension not in ['.jpg', '.jpeg', '.png']:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file format. Please upload JPG or PNG images."
                )
        
        result = SignatureProcessor.compare_two_signatures(signature1, signature2)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing signatures: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_version": "1.0.0",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    init_db()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)