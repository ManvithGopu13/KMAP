"""
FastAPI service for ML model
Provides REST API endpoints for subscription detection
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import logging

from data_preprocessing import TransactionPreprocessor
from subscription_detector import SubscriptionDetector
from pdf_parser import PDFParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Subscription Detection API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
preprocessor = TransactionPreprocessor()
detector = SubscriptionDetector()
pdf_parser = PDFParser()


# Request/Response Models
class Transaction(BaseModel):
    date: str
    amount: float
    description: str
    type: str = "debit"


class TransactionList(BaseModel):
    transactions: List[Transaction]


class Subscription(BaseModel):
    merchant_name: str
    amount: float
    frequency: str
    category: str
    first_charged: str
    last_charged: str
    next_billing_date: str
    transaction_count: int
    confidence_score: float
    annual_cost: float
    status: str


class SubscriptionResponse(BaseModel):
    success: bool
    subscriptions: List[Subscription]
    total_monthly_cost: float
    total_annual_cost: float
    message: Optional[str] = None


class PDFCheckResponse(BaseModel):
    is_encrypted: bool
    requires_password: bool


@app.get("/")
def root():
    return {
        "service": "Subscription Detection API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/check-pdf")
async def check_pdf(file: UploadFile = File(...)):
    """
    Check if PDF is password protected
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Check encryption
        is_encrypted = pdf_parser.check_if_encrypted(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return PDFCheckResponse(
            is_encrypted=is_encrypted,
            requires_password=is_encrypted
        )
        
    except Exception as e:
        logger.error(f"Error checking PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse-pdf")
async def parse_pdf(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None)
):
    """
    Parse PDF and extract transactions
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Parse PDF
        success, transactions, error = pdf_parser.parse_bank_statement(
            tmp_path, 
            password
        )
        
        # Clean up
        os.unlink(tmp_path)
        
        if not success:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "success": True,
            "transaction_count": len(transactions),
            "transactions": transactions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-subscriptions", response_model=SubscriptionResponse)
async def detect_subscriptions(data: TransactionList):
    """
    Detect subscriptions from transaction list
    """
    try:
        transactions = [txn.dict() for txn in data.transactions]
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions provided")
        
        # Preprocess
        df = preprocessor.load_transactions(transactions)
        df = preprocessor.clean_transactions(df)
        df = preprocessor.add_features(df)
        grouped = preprocessor.group_by_merchant(df)
        
        # Detect
        subscriptions = detector.detect_subscriptions(grouped, confidence_threshold=0.6)
        
        # Calculate totals
        total_monthly = sum(
            sub['amount'] for sub in subscriptions 
            if sub['frequency'] == 'monthly'
        )
        total_annual = sum(sub['annual_cost'] for sub in subscriptions)
        
        return SubscriptionResponse(
            success=True,
            subscriptions=subscriptions,
            total_monthly_cost=round(total_monthly, 2),
            total_annual_cost=round(total_annual, 2),
            message=f"Found {len(subscriptions)} subscriptions"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting subscriptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-pdf-full", response_model=SubscriptionResponse)
async def process_pdf_full(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None)
):
    """
    Complete pipeline: Parse PDF + Detect Subscriptions
    """
    try:
        # Save file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Parse
        success, transactions, error = pdf_parser.parse_bank_statement(
            tmp_path,
            password
        )
        
        os.unlink(tmp_path)
        
        if not success:
            raise HTTPException(status_code=400, detail=error)
        
        # Detect subscriptions
        df = preprocessor.load_transactions(transactions)
        df = preprocessor.clean_transactions(df)
        df = preprocessor.add_features(df)
        grouped = preprocessor.group_by_merchant(df)
        
        subscriptions = detector.detect_subscriptions(grouped, confidence_threshold=0.6)
        
        # Calculate totals
        total_monthly = sum(
            sub['amount'] for sub in subscriptions 
            if sub['frequency'] == 'monthly'
        )
        total_annual = sum(sub['annual_cost'] for sub in subscriptions)
        
        return SubscriptionResponse(
            success=True,
            subscriptions=subscriptions,
            total_monthly_cost=round(total_monthly, 2),
            total_annual_cost=round(total_annual, 2),
            message=f"Found {len(subscriptions)} subscriptions from {len(transactions)} transactions"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in full pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)