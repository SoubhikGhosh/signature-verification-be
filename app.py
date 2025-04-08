from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import os
import logging
import time
from typing import List, Dict, Any
import google.generativeai as genai
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

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

# Configure the Gemini API
logger.info("Configuring Gemini API")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="Signature Similarity API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

class SimilarityResponse(BaseModel):
    similarity_score: float
    analysis: str

@app.post("/compare-signatures/", response_model=SimilarityResponse)
async def compare_signatures(
    signature1: UploadFile = File(...),
    signature2: UploadFile = File(...),
):
    """
    Compare two signature images and return a similarity score.
    """
    logger.info(f"Received signature comparison request: {signature1.filename} vs {signature2.filename}")
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
You are tasked with acting as an expert system for forensic handwriting comparison. Your objective is to perform a detailed comparative analysis of two signature specimens provided as images:
*   Specimen A: [Signature Image 1 Placeholder]
*   Specimen B: [Signature Image 2 Placeholder]

Conduct a systematic examination based on forensic document examination principles to assess the likelihood that both signatures were executed by the same individual.

**Methodology - Focus Areas:** Evaluate and compare the following aspects meticulously:

1.  **Macro-Features (Overall Impression):**
    *   **Arrangement & Placement:** Position on the page/line (if context available).
    *   **Size & Proportions:** Overall dimensions, relative height of capitals vs. lowercase, internal proportions within letters.
    *   **Slant:** General angle of inclination.
    *   **Spacing:** Between letters, components, and potentially words if present.
    *   **Overall Pictorial Effect:** General shape, style (e.g., cursive, printed, mixed), and aesthetic appearance.

2.  **Micro-Features (Detailed Execution):**
    *   **Movement & Line Quality:** Assess rhythm, speed (implied by smoothness/tapering vs. hesitation/blunt ends), fluency, and coordination. Look for tremors, abrupt angle changes, or patching/retouching.
    *   **Pressure Patterns:** Identify characteristic variations in line width indicating where pressure is typically applied or released.
    *   **Letter Formation:** Detailed construction of specific characters (allographs), including loops (open/closed, size, shape), initial/terminal strokes, 'i' dots, 't' crosses (position, length, shape), connections between letters (garlanded, angular, threaded), and any idiosyncratic features or embellishments.
    *   **Baseline Habits:** Alignment relative to a real or imaginary baseline (on-line, above, below, undulating).

**Analytical Considerations:**

*   **Range of Variation:** Consider the concept of natural variation. Genuine signatures from the same person will exhibit some differences. Your analysis must differentiate these expected variations from significant, fundamental discrepancies in writing habits.
*   **Consistency vs. Discrepancy:** Weigh the similarities against the differences. Are the similarities common (class characteristics) or unique (individual characteristics)? Are the differences minor variations or fundamental divergences?
*   **Image Quality Assessment:** Briefly comment if the resolution, clarity, or angle of the provided images significantly hinders a thorough analysis.
*   **Handling Non-Signatures:** If either image clearly does not depict a handwritten signature, state this and decline to provide a score.

**Output Requirements:**

1.  **Similarity Score:** Assign a numerical score between 0.00 and 1.00.
    *   `0.00`: Indicates fundamental, irreconcilable differences suggesting different authorship.
    *   `0.50`: Indicates a mix of significant similarities and differences, leading to an inconclusive comparison.
    *   `1.00`: Indicates a high degree of consistency in significant writing habits, strongly suggesting common authorship (allowing for natural variation). *Note: Absolute identicalness (pixel-level) is highly suspicious and should not typically result in a 1.00 score in a forensic context.*

2.  **Analysis Rationale:** Provide a concise, objective summary justifying the assigned score. This summary MUST briefly reference specific findings from the Methodology focus areas (both macro and micro features), highlighting key points of consistency and/or discrepancy that led to your conclusion.

**Format:** Adhere strictly to the following output format:

Score: [similarity score as a decimal between 0.00 and 1.00]
Analysis: [Concise justification referencing specific macro and micro features, explaining the basis for the score.]
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
        
        response_data = SimilarityResponse(
            similarity_score=similarity_score,
            analysis=analysis
        )
        logger.info("Successfully completed signature comparison")
        return response_data
            
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }
    
    return health_data

# Track application startup time
startup_time = time.time()
logger.info("Application initialized")

if __name__ == "__main__":
    logger.info("Starting Signature Similarity API server")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)