"""
Data preprocessing module for transaction data
Cleans, normalizes, and prepares data for ML model
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionPreprocessor:
    """
    Preprocesses transaction data from bank statements
    """
    
    def __init__(self):
        self.common_payment_patterns = [
            r'NEFT|IMPS|UPI|RTGS|CARD',
            r'DEBIT|CREDIT',
            r'AUTO',
        ]
        
        # Common subscription merchants (will be expanded)
        self.known_merchants = {
            'netflix': 'streaming',
            'amazon prime': 'streaming',
            'spotify': 'music',
            'disney': 'streaming',
            'hotstar': 'streaming',
            'youtube': 'streaming',
            'swiggy': 'food_delivery',
            'zomato': 'food_delivery',
            'uber': 'transportation',
            'ola': 'transportation',
            'phonepe': 'digital_wallet',
            'paytm': 'digital_wallet',
            'google': 'software',
            'microsoft': 'software',
            'adobe': 'software',
            'linkedin': 'professional',
            'gym': 'fitness',
            'electricity': 'utility',
            'water': 'utility',
            'gas': 'utility',
            'broadband': 'utility',
            'internet': 'utility',
            'insurance': 'insurance',
            'emi': 'loan',
        }
    
    def load_transactions(self, data: List[Dict]) -> pd.DataFrame:
        """
        Load transactions into DataFrame
        
        Args:
            data: List of transaction dictionaries
        
        Returns:
            DataFrame with transactions
        """
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} transactions")
        return df
    
    def clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize transaction data
        """
        df = df.copy()
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert amount to float
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            # Remove negative amounts (keep only debits for subscription detection)
            df = df[df['amount'] > 0]
        
        # Clean description/narration
        if 'description' in df.columns:
            df['description'] = df['description'].fillna('')
            df['description_clean'] = df['description'].apply(self._clean_text)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'amount', 'description'], keep='first')
        
        # Sort by date
        df = df.sort_values('date')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"Cleaned data: {len(df)} transactions remaining")
        return df
    
    def _clean_text(self, text: str) -> str:
        """
        Clean transaction description text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_merchant_name(self, description: str) -> Optional[str]:
        """
        Extract merchant name from transaction description
        """
        description = description.lower()
        
        # Check against known merchants
        for merchant, category in self.known_merchants.items():
            if merchant in description:
                return merchant
        
        # Try to extract merchant name using patterns
        # UPI pattern: netflix.com@icici or something@paytm
        upi_pattern = r'(\w+)\.?\w*@\w+'
        match = re.search(upi_pattern, description)
        if match:
            return match.group(1)
        
        # Card payment pattern: POS <merchant name>
        pos_pattern = r'pos\s+([a-z\s]+)'
        match = re.search(pos_pattern, description)
        if match:
            return match.group(1).strip()
        
        # Return first meaningful word (longer than 3 chars)
        words = description.split()
        for word in words:
            if len(word) > 3 and word not in ['neft', 'imps', 'card', 'debit', 'credit']:
                return word
        
        return None
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional features for ML model
        """
        df = df.copy()
        
        # Extract merchant name
        df['merchant'] = df['description_clean'].apply(self.extract_merchant_name)
        
        # Add day of month, day of week
        df['day_of_month'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # Add amount range category
        df['amount_range'] = pd.cut(
            df['amount'],
            bins=[0, 100, 500, 1000, 5000, float('inf')],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        logger.info("Added features to transactions")
        return df
    
    def group_by_merchant(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group transactions by merchant
        """
        grouped = {}
        
        for merchant in df['merchant'].unique():
            if pd.notna(merchant):
                merchant_df = df[df['merchant'] == merchant].copy()
                if len(merchant_df) >= 2:  # At least 2 transactions for pattern
                    grouped[merchant] = merchant_df
        
        logger.info(f"Grouped into {len(grouped)} merchant groups")
        return grouped


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_transactions = [
        {
            'date': '2024-01-15',
            'amount': 799.0,
            'description': 'UPI/NETFLIX.COM@ICICI/Payment',
            'type': 'debit'
        },
        {
            'date': '2024-02-15',
            'amount': 799.0,
            'description': 'UPI/NETFLIX.COM@ICICI/Payment',
            'type': 'debit'
        },
        {
            'date': '2024-03-15',
            'amount': 799.0,
            'description': 'UPI/NETFLIX.COM@ICICI/Payment',
            'type': 'debit'
        },
        {
            'date': '2024-01-20',
            'amount': 119.0,
            'description': 'UPI/SPOTIFY@PAYTM/Subscription',
            'type': 'debit'
        },
        {
            'date': '2024-02-20',
            'amount': 119.0,
            'description': 'UPI/SPOTIFY@PAYTM/Subscription',
            'type': 'debit'
        },
    ]
    
    preprocessor = TransactionPreprocessor()
    df = preprocessor.load_transactions(sample_transactions)
    df = preprocessor.clean_transactions(df)
    df = preprocessor.add_features(df)
    
    print(df[['date', 'amount', 'merchant', 'day_of_month']])