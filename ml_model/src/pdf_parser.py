"""
PDF parser for extracting transaction data from bank statements
Handles both password-protected and open PDFs
"""

import pdfplumber
import pandas as pd
import re
from typing import List, Dict, Optional, Tuple
from pypdf import PdfReader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFParser:
    """
    Extract transaction data from bank statement PDFs
    """
    
    def __init__(self):
        # Common date patterns in bank statements
        self.date_patterns = [
            r'\d{2}[-/]\d{2}[-/]\d{4}',  # DD-MM-YYYY or DD/MM/YYYY
            r'\d{4}[-/]\d{2}[-/]\d{2}',  # YYYY-MM-DD
            r'\d{2}\s+[A-Za-z]{3}\s+\d{4}',  # DD Mon YYYY
        ]
        
        # Amount pattern
        self.amount_pattern = r'[\d,]+\.\d{2}'
        
    def check_if_encrypted(self, pdf_path: str) -> bool:
        """
        Check if PDF is password protected
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if encrypted, False otherwise
        """
        try:
            reader = PdfReader(pdf_path)
            return reader.is_encrypted
        except Exception as e:
            logger.error(f"Error checking PDF encryption: {e}")
            return False
    
    def decrypt_pdf(self, pdf_path: str, password: str) -> bool:
        """
        Try to decrypt PDF with provided password
        
        Args:
            pdf_path: Path to PDF file
            password: Password to try
            
        Returns:
            True if successful, False otherwise
        """
        try:
            reader = PdfReader(pdf_path)
            if reader.is_encrypted:
                result = reader.decrypt(password)
                # Result: 0 = failed, 1 = user password, 2 = owner password
                if result > 0:
                    logger.info("PDF decrypted successfully")
                    return True
                else:
                    logger.warning("Password incorrect")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error decrypting PDF: {e}")
            return False
    
    def extract_text_from_pdf(
        self, 
        pdf_path: str, 
        password: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Extract text from PDF (handles password protection)
        
        Args:
            pdf_path: Path to PDF file
            password: Password if PDF is encrypted
            
        Returns:
            (success, extracted_text)
        """
        try:
            # Check if encrypted
            if self.check_if_encrypted(pdf_path):
                if not password:
                    return (False, "PDF is password protected. Please provide password.")
                
                # Try to decrypt
                if not self.decrypt_pdf(pdf_path, password):
                    return (False, "Incorrect password")
            
            # Extract text using pdfplumber
            text = ""
            with pdfplumber.open(pdf_path, password=password) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return (True, text)
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return (False, f"Error: {str(e)}")
    
    def extract_tables_from_pdf(
        self, 
        pdf_path: str,
        password: Optional[str] = None
    ) -> Tuple[bool, List[pd.DataFrame]]:
        """
        Extract tables from PDF (better for structured statements)
        
        Args:
            pdf_path: Path to PDF file
            password: Password if encrypted
            
        Returns:
            (success, list of DataFrames)
        """
        try:
            if self.check_if_encrypted(pdf_path) and not password:
                return (False, [])
            
            tables = []
            with pdfplumber.open(pdf_path, password=password) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:  # Has header + data
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df['page_number'] = page_num + 1
                            tables.append(df)
            
            logger.info(f"Extracted {len(tables)} tables from PDF")
            return (True, tables)
            
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return (False, [])
    
    def parse_transactions_from_text(self, text: str) -> List[Dict]:
        """
        Parse transactions from extracted text
        Uses pattern matching to identify transaction lines
        """
        transactions = []
        lines = text.split('\n')
        
        for line in lines:
            # Try to find date, amount, and description
            date_match = None
            for pattern in self.date_patterns:
                date_match = re.search(pattern, line)
                if date_match:
                    break
            
            if not date_match:
                continue
            
            # Find amounts
            amounts = re.findall(self.amount_pattern, line)
            if not amounts:
                continue
            
            # Extract date
            date_str = date_match.group()
            
            # Extract amount (usually the debit amount)
            amount_str = amounts[0] if len(amounts) > 0 else None
            
            # Extract description (text before/after date)
            description = line.strip()
            
            if date_str and amount_str:
                transactions.append({
                    'date': date_str,
                    'amount': float(amount_str.replace(',', '')),
                    'description': description,
                    'type': 'debit'  # Assume debit for subscriptions
                })
        
        logger.info(f"Parsed {len(transactions)} transactions from text")
        return transactions
    
    def parse_transactions_from_tables(
        self, 
        tables: List[pd.DataFrame]
    ) -> List[Dict]:
        """
        Parse transactions from extracted tables
        More reliable than text parsing
        """
        all_transactions = []
        
        for table in tables:
            # Try to identify columns
            columns = table.columns.tolist()
            
            # Common column names (case-insensitive)
            date_col = self._find_column(columns, ['date', 'txn date', 'transaction date'])
            amount_col = self._find_column(columns, ['amount', 'debit', 'withdrawal', 'paid'])
            desc_col = self._find_column(columns, ['description', 'narration', 'particulars', 'details'])
            
            if not (date_col and amount_col):
                continue
            
            for _, row in table.iterrows():
                try:
                    date_val = row[date_col]
                    amount_val = row[amount_col]
                    desc_val = row[desc_col] if desc_col else ''
                    
                    # Clean amount
                    if isinstance(amount_val, str):
                        amount_val = amount_val.replace(',', '').strip()
                        if not amount_val:
                            continue
                        amount_val = float(amount_val)
                    
                    if pd.notna(date_val) and pd.notna(amount_val) and amount_val > 0:
                        all_transactions.append({
                            'date': str(date_val),
                            'amount': float(amount_val),
                            'description': str(desc_val),
                            'type': 'debit'
                        })
                except Exception as e:
                    continue
        
        logger.info(f"Parsed {len(all_transactions)} transactions from tables")
        return all_transactions
    
    def _find_column(self, columns: List[str], possible_names: List[str]) -> Optional[str]:
        """
        Find column by matching possible names (case-insensitive)
        """
        for col in columns:
            col_lower = str(col).lower().strip()
            for name in possible_names:
                if name in col_lower:
                    return col
        return None
    
    def parse_bank_statement(
        self, 
        pdf_path: str,
        password: Optional[str] = None,
        use_tables: bool = True
    ) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Main method to parse bank statement PDF
        
        Args:
            pdf_path: Path to PDF
            password: Password if encrypted
            use_tables: Try table extraction first (more reliable)
            
        Returns:
            (success, transactions, error_message)
        """
        try:
            # Check encryption
            is_encrypted = self.check_if_encrypted(pdf_path)
            
            if is_encrypted and not password:
                return (False, [], "PDF is password protected. Please provide password.")
            
            # Try table extraction first
            if use_tables:
                success, tables = self.extract_tables_from_pdf(pdf_path, password)
                if success and tables:
                    transactions = self.parse_transactions_from_tables(tables)
                    if transactions:
                        return (True, transactions, None)
            
            # Fallback to text extraction
            success, text = self.extract_text_from_pdf(pdf_path, password)
            if not success:
                return (False, [], text)
            
            transactions = self.parse_transactions_from_text(text)
            
            if not transactions:
                return (False, [], "Could not extract transactions from PDF. Please check format.")
            
            return (True, transactions, None)
            
        except Exception as e:
            logger.error(f"Error parsing bank statement: {e}")
            return (False, [], str(e))


# Example usage
if __name__ == "__main__":
    parser = PDFParser()
    
    # Test with sample PDF
    pdf_path = "1211_01012026135632.pdf"
    
    # Check if encrypted
    if parser.check_if_encrypted(pdf_path):
        print("PDF is password protected")
        password = input("Enter password: ")
    else:
        password = None
    
    # Parse
    success, transactions, error = parser.parse_bank_statement(pdf_path, password)
    
    if success:
        print(f"\nExtracted {len(transactions)} transactions:")
        for txn in transactions[:5]:  # Show first 5
            print(f"  {txn['date']}: ₹{txn['amount']} - {txn['description'][:50]}")
    else:
        print(f"Error: {error}")