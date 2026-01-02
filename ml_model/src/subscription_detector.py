"""
ML model for detecting subscriptions from transaction patterns
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubscriptionDetector:
    """
    Detects recurring payments and subscriptions from transaction data
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.patterns = {
            'weekly': {'min_days': 5, 'max_days': 9, 'tolerance': 2},
            'biweekly': {'min_days': 12, 'max_days': 16, 'tolerance': 2},
            'monthly': {'min_days': 25, 'max_days': 35, 'tolerance': 5},
            'quarterly': {'min_days': 80, 'max_days': 100, 'tolerance': 10},
            'annual': {'min_days': 350, 'max_days': 380, 'tolerance': 15},
        }
    
    def detect_subscriptions(
        self, 
        grouped_transactions: Dict[str, pd.DataFrame],
        confidence_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Detect subscriptions from grouped transactions
        
        Args:
            grouped_transactions: Transactions grouped by merchant
            confidence_threshold: Minimum confidence score (0-1)
        
        Returns:
            List of detected subscriptions
        """
        subscriptions = []
        
        for merchant, transactions in grouped_transactions.items():
            if len(transactions) < 2:
                continue
            
            # Sort by date
            transactions = transactions.sort_values('date')
            
            # Calculate intervals between transactions
            intervals = self._calculate_intervals(transactions)
            
            # Check if pattern is recurring
            pattern_result = self._detect_pattern(intervals, transactions)
            
            if pattern_result['is_recurring']:
                subscription = self._create_subscription(
                    merchant,
                    transactions,
                    intervals,
                    pattern_result
                )
                
                if subscription['confidence_score'] >= confidence_threshold:
                    subscriptions.append(subscription)
                    logger.info(f"Detected subscription: {merchant} - {subscription['frequency']}")
        
        logger.info(f"Total subscriptions detected: {len(subscriptions)}")
        return subscriptions
    
    def _calculate_intervals(self, transactions: pd.DataFrame) -> List[int]:
        """
        Calculate days between consecutive transactions
        """
        intervals = []
        dates = transactions['date'].tolist()
        
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i-1]).days
            intervals.append(delta)
        
        return intervals
    
    def _detect_pattern(
        self, 
        intervals: List[int], 
        transactions: pd.DataFrame
    ) -> Dict:
        """
        Detect if intervals follow a recurring pattern
        """
        if not intervals:
            return {'is_recurring': False}
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Check against known patterns
        detected_frequency = None
        for freq_name, freq_params in self.patterns.items():
            if freq_params['min_days'] <= avg_interval <= freq_params['max_days']:
                # Check if standard deviation is within tolerance
                if std_interval <= freq_params['tolerance']:
                    detected_frequency = freq_name
                    break
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            intervals,
            transactions,
            detected_frequency
        )
        
        return {
            'is_recurring': detected_frequency is not None,
            'frequency': detected_frequency,
            'avg_interval': avg_interval,
            'std_interval': std_interval,
            'confidence': confidence
        }
    
    def _calculate_confidence(
        self,
        intervals: List[int],
        transactions: pd.DataFrame,
        frequency: Optional[str]
    ) -> float:
        """
        Calculate confidence score for subscription detection
        """
        if not frequency:
            return 0.0
        
        scores = []
        
        # 1. Interval consistency (40% weight)
        std_interval = np.std(intervals)
        avg_interval = np.mean(intervals)
        if avg_interval > 0:
            cv = std_interval / avg_interval  # Coefficient of variation
            interval_score = max(0, 1 - cv)
            scores.append(interval_score * 0.4)
        
        # 2. Amount consistency (30% weight)
        amounts = transactions['amount'].tolist()
        std_amount = np.std(amounts)
        avg_amount = np.mean(amounts)
        if avg_amount > 0:
            cv_amount = std_amount / avg_amount
            amount_score = max(0, 1 - cv_amount)
            scores.append(amount_score * 0.3)
        
        # 3. Number of occurrences (20% weight)
        num_transactions = len(transactions)
        occurrence_score = min(1.0, num_transactions / 6)  # 6+ transactions = max score
        scores.append(occurrence_score * 0.2)
        
        # 4. Recency (10% weight)
        last_date = transactions['date'].max()
        days_since_last = (datetime.now() - last_date).days
        recency_score = max(0, 1 - (days_since_last / 60))  # 60 days threshold
        scores.append(recency_score * 0.1)
        
        total_confidence = sum(scores)
        return round(total_confidence, 2)
    
    def _create_subscription(
        self,
        merchant: str,
        transactions: pd.DataFrame,
        intervals: List[int],
        pattern_result: Dict
    ) -> Dict:
        """
        Create subscription object from detected pattern
        """
        amounts = transactions['amount'].tolist()
        dates = transactions['date'].tolist()
        
        # Calculate average amount
        avg_amount = np.mean(amounts)
        
        # Predict next billing date
        avg_interval = pattern_result['avg_interval']
        last_date = dates[-1]
        next_date = last_date + timedelta(days=int(avg_interval))
        
        # Calculate annual cost
        annual_multiplier = {
            'weekly': 52,
            'biweekly': 26,
            'monthly': 12,
            'quarterly': 4,
            'annual': 1
        }
        frequency = pattern_result['frequency']
        annual_cost = avg_amount * annual_multiplier.get(frequency, 12)
        
        # Determine category (basic classification)
        category = self._classify_category(merchant)
        
        subscription = {
            'merchant_name': merchant.title(),
            'amount': round(avg_amount, 2),
            'frequency': frequency,
            'category': category,
            'first_charged': dates[0].strftime('%Y-%m-%d'),
            'last_charged': dates[-1].strftime('%Y-%m-%d'),
            'next_billing_date': next_date.strftime('%Y-%m-%d'),
            'transaction_count': len(transactions),
            'confidence_score': pattern_result['confidence'],
            'annual_cost': round(annual_cost, 2),
            'status': 'active' if (datetime.now() - last_date).days < 60 else 'inactive',
            'avg_interval_days': int(avg_interval),
            'amount_variance': round(np.std(amounts), 2)
        }
        
        return subscription
    
    def _classify_category(self, merchant: str) -> str:
        """
        Classify merchant into category
        """
        categories = {
            'streaming': ['netflix', 'prime', 'disney', 'hotstar', 'youtube', 'spotify', 'apple music'],
            'software': ['adobe', 'microsoft', 'google', 'dropbox', 'zoom'],
            'fitness': ['gym', 'fitness', 'cult', 'healthify'],
            'food': ['swiggy', 'zomato', 'dunzo'],
            'transportation': ['uber', 'ola', 'rapido'],
            'utility': ['electricity', 'water', 'gas', 'broadband', 'internet'],
            'insurance': ['insurance', 'lic', 'hdfc life'],
            'loan': ['emi', 'loan', 'credit'],
        }
        
        merchant_lower = merchant.lower()
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in merchant_lower:
                    return category
        
        return 'other'
    
    def train_model(self, training_data: pd.DataFrame):
        """
        Train ML model on labeled data (for future enhancement)
        """
        # This would be used when we have labeled training data
        # For now, we use rule-based detection
        logger.info("Training ML model...")
        # Implementation for supervised learning
        pass
    
    def save_model(self, filepath: str):
        """
        Save trained model to file
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'patterns': self.patterns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.patterns = model_data['patterns']
        logger.info(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    from data_preprocessing import TransactionPreprocessor
    
    # Sample transactions
    sample_data = [
        {'date': '2024-01-15', 'amount': 799.0, 'description': 'NETFLIX'},
        {'date': '2024-02-15', 'amount': 799.0, 'description': 'NETFLIX'},
        {'date': '2024-03-15', 'amount': 799.0, 'description': 'NETFLIX'},
        {'date': '2024-04-15', 'amount': 799.0, 'description': 'NETFLIX'},
        {'date': '2024-01-20', 'amount': 119.0, 'description': 'SPOTIFY'},
        {'date': '2024-02-20', 'amount': 119.0, 'description': 'SPOTIFY'},
        {'date': '2024-03-20', 'amount': 119.0, 'description': 'SPOTIFY'},
    ]
    
    # Preprocess
    preprocessor = TransactionPreprocessor()
    df = preprocessor.load_transactions(sample_data)
    df = preprocessor.clean_transactions(df)
    df = preprocessor.add_features(df)
    grouped = preprocessor.group_by_merchant(df)
    
    # Detect subscriptions
    detector = SubscriptionDetector()
    subscriptions = detector.detect_subscriptions(grouped)
    
    # Print results
    for sub in subscriptions:
        print(f"\n{sub['merchant_name']}:")
        print(f"  Amount: ₹{sub['amount']}")
        print(f"  Frequency: {sub['frequency']}")
        print(f"  Next billing: {sub['next_billing_date']}")
        print(f"  Confidence: {sub['confidence_score']}")
        print(f"  Annual cost: ₹{sub['annual_cost']}")