"""
Smart Data Validation and Quality Assurance Engine
Advanced data integrity checking and quality metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Represents the result of a data validation check"""
    check_name: str
    status: str  # 'pass', 'warning', 'fail'
    message: str
    affected_rows: int
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendations: List[str]
    data_sample: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None

@dataclass
class QualityMetrics:
    """Data quality metrics summary"""
    overall_score: float  # 0-100
    completeness_score: float
    validity_score: float
    consistency_score: float
    uniqueness_score: float
    total_records: int
    issues_found: int
    critical_issues: int
    warnings: int
    timestamp: datetime

class DataQualityValidator:
    """Advanced data validation and quality assurance engine"""
    
    def __init__(self):
        """Initialize the data quality validator"""
        self.validation_rules = self._initialize_validation_rules()
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.validation_history = []
        self.data_profiling_cache = {}
        
        logger.info("Data Quality Validator initialized")
    
    def _initialize_validation_rules(self) -> Dict[str, Dict]:
        """Initialize validation rules for different data types"""
        return {
            'rating': {
                'type': 'numeric',
                'min_value': 1.0,
                'max_value': 5.0,
                'required': True,
                'description': 'Product rating should be between 1-5'
            },
            'review_text': {
                'type': 'text',
                'min_length': 10,
                'max_length': 10000,
                'required': True,
                'description': 'Review text should be meaningful and within length limits'
            },
            'product': {
                'type': 'categorical',
                'required': True,
                'description': 'Product name should be consistent and non-empty'
            },
            'brand': {
                'type': 'categorical',
                'required': False,
                'description': 'Brand name should follow consistent naming conventions'
            },
            'sentiment_label': {
                'type': 'categorical',
                'allowed_values': ['positive', 'negative', 'neutral'],
                'required': False,
                'description': 'Sentiment should be one of: positive, negative, neutral'
            },
            'timestamp': {
                'type': 'datetime',
                'required': False,
                'min_date': '2010-01-01',
                'max_date': None,  # Will be set to current date
                'description': 'Review timestamp should be realistic'
            }
        }
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality score thresholds"""
        return {
            'excellent': 95.0,
            'good': 85.0,
            'fair': 70.0,
            'poor': 50.0,
            'critical': 30.0
        }
    
    def validate_dataset(self, df: pd.DataFrame, detailed: bool = True) -> Tuple[List[ValidationResult], QualityMetrics]:
        """
        Comprehensive dataset validation
        
        Args:
            df: DataFrame to validate
            detailed: Whether to include detailed validation results
            
        Returns:
            Tuple of validation results and quality metrics
        """
        validation_results = []
        
        try:
            if df is None or df.empty:
                return [ValidationResult(
                    check_name="empty_dataset",
                    status="fail",
                    message="Dataset is empty or None",
                    affected_rows=0,
                    severity="critical",
                    recommendations=["Provide a valid dataset with review data"]
                )], self._create_empty_metrics()
            
            logger.info(f"Starting validation of dataset with {len(df)} records")
            
            # Basic structure validation
            validation_results.extend(self._validate_basic_structure(df))
            
            # Column-specific validation
            validation_results.extend(self._validate_columns(df))
            
            # Data consistency validation
            validation_results.extend(self._validate_consistency(df))
            
            # Duplicate detection
            validation_results.extend(self._validate_duplicates(df))
            
            # Statistical validation
            validation_results.extend(self._validate_statistics(df))
            
            # Business logic validation
            validation_results.extend(self._validate_business_rules(df))
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(df, validation_results)
            
            # Store validation history
            self.validation_history.append({
                'timestamp': datetime.now(),
                'results': validation_results,
                'metrics': quality_metrics
            })
            
            logger.info(f"Validation completed. Overall quality score: {quality_metrics.overall_score:.1f}")
            
            return validation_results, quality_metrics
            
        except Exception as e:
            logger.error(f"Error during dataset validation: {e}")
            return [ValidationResult(
                check_name="validation_error",
                status="fail",
                message=f"Validation failed due to error: {str(e)}",
                affected_rows=0,
                severity="critical",
                recommendations=["Check dataset format and try again"]
            )], self._create_empty_metrics()
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate basic dataset structure"""
        results = []
        
        # Check if dataset has minimum required rows
        min_rows = 10
        if len(df) < min_rows:
            results.append(ValidationResult(
                check_name="insufficient_data",
                status="warning",
                message=f"Dataset has only {len(df)} rows, minimum recommended is {min_rows}",
                affected_rows=len(df),
                severity="medium",
                recommendations=["Collect more review data for reliable analysis"]
            ))
        
        # Check for essential columns
        essential_columns = ['product', 'review_text']
        missing_columns = [col for col in essential_columns if col not in df.columns]
        
        if missing_columns:
            results.append(ValidationResult(
                check_name="missing_essential_columns",
                status="fail",
                message=f"Missing essential columns: {missing_columns}",
                affected_rows=len(df),
                severity="critical",
                recommendations=[f"Add missing columns: {', '.join(missing_columns)}"]
            ))
        
        # Check for completely empty columns
        empty_columns = [col for col in df.columns if df[col].isnull().all()]
        if empty_columns:
            results.append(ValidationResult(
                check_name="empty_columns",
                status="warning",
                message=f"Completely empty columns found: {empty_columns}",
                affected_rows=len(df),
                severity="low",
                recommendations=["Remove empty columns or populate with data"]
            ))
        
        return results
    
    def _validate_columns(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate individual columns based on rules"""
        results = []
        
        for column, rules in self.validation_rules.items():
            if column not in df.columns:
                if rules.get('required', False):
                    results.append(ValidationResult(
                        check_name=f"missing_required_column_{column}",
                        status="fail",
                        message=f"Required column '{column}' is missing",
                        affected_rows=len(df),
                        severity="high",
                        recommendations=[f"Add '{column}' column with valid data"]
                    ))
                continue
            
            # Validate column data
            column_results = self._validate_single_column(df, column, rules)
            results.extend(column_results)
        
        return results
    
    def _validate_single_column(self, df: pd.DataFrame, column: str, rules: Dict) -> List[ValidationResult]:
        """Validate a single column based on its rules"""
        results = []
        series = df[column]
        
        # Null value validation
        null_count = series.isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        
        if rules.get('required', False) and null_count > 0:
            results.append(ValidationResult(
                check_name=f"null_values_{column}",
                status="warning" if null_percentage < 10 else "fail",
                message=f"'{column}' has {null_count} null values ({null_percentage:.1f}%)",
                affected_rows=null_count,
                severity="medium" if null_percentage < 10 else "high",
                recommendations=[f"Fill null values in '{column}' column"]
            ))
        
        # Type-specific validation
        valid_data = series.dropna()
        
        if rules['type'] == 'numeric':
            results.extend(self._validate_numeric_column(valid_data, column, rules))
        elif rules['type'] == 'text':
            results.extend(self._validate_text_column(valid_data, column, rules))
        elif rules['type'] == 'categorical':
            results.extend(self._validate_categorical_column(valid_data, column, rules))
        elif rules['type'] == 'datetime':
            results.extend(self._validate_datetime_column(valid_data, column, rules))
        
        return results
    
    def _validate_numeric_column(self, series: pd.Series, column: str, rules: Dict) -> List[ValidationResult]:
        """Validate numeric column"""
        results = []
        
        # Check if values are numeric
        non_numeric = series.apply(lambda x: not isinstance(x, (int, float, np.integer, np.floating)))
        non_numeric_count = non_numeric.sum()
        
        if non_numeric_count > 0:
            results.append(ValidationResult(
                check_name=f"non_numeric_values_{column}",
                status="fail",
                message=f"'{column}' contains {non_numeric_count} non-numeric values",
                affected_rows=non_numeric_count,
                severity="high",
                recommendations=[f"Convert non-numeric values in '{column}' to proper numeric format"]
            ))
        
        # Range validation
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if 'min_value' in rules:
            below_min = (numeric_series < rules['min_value']).sum()
            if below_min > 0:
                results.append(ValidationResult(
                    check_name=f"below_minimum_{column}",
                    status="fail",
                    message=f"'{column}' has {below_min} values below minimum ({rules['min_value']})",
                    affected_rows=below_min,
                    severity="high",
                    recommendations=[f"Ensure all '{column}' values are >= {rules['min_value']}"]
                ))
        
        if 'max_value' in rules:
            above_max = (numeric_series > rules['max_value']).sum()
            if above_max > 0:
                results.append(ValidationResult(
                    check_name=f"above_maximum_{column}",
                    status="fail",
                    message=f"'{column}' has {above_max} values above maximum ({rules['max_value']})",
                    affected_rows=above_max,
                    severity="high",
                    recommendations=[f"Ensure all '{column}' values are <= {rules['max_value']}"]
                ))
        
        return results
    
    def _validate_text_column(self, series: pd.Series, column: str, rules: Dict) -> List[ValidationResult]:
        """Validate text column"""
        results = []
        
        # Convert to string and check lengths
        text_series = series.astype(str)
        lengths = text_series.str.len()
        
        if 'min_length' in rules:
            too_short = (lengths < rules['min_length']).sum()
            if too_short > 0:
                results.append(ValidationResult(
                    check_name=f"text_too_short_{column}",
                    status="warning",
                    message=f"'{column}' has {too_short} values shorter than {rules['min_length']} characters",
                    affected_rows=too_short,
                    severity="medium",
                    recommendations=[f"Review short entries in '{column}' column"]
                ))
        
        if 'max_length' in rules:
            too_long = (lengths > rules['max_length']).sum()
            if too_long > 0:
                results.append(ValidationResult(
                    check_name=f"text_too_long_{column}",
                    status="warning",
                    message=f"'{column}' has {too_long} values longer than {rules['max_length']} characters",
                    affected_rows=too_long,
                    severity="low",
                    recommendations=[f"Consider truncating very long entries in '{column}'"]
                ))
        
        # Check for suspicious patterns
        if column == 'review_text':
            suspicious_patterns = self._detect_suspicious_text_patterns(text_series)
            results.extend(suspicious_patterns)
        
        return results
    
    def _validate_categorical_column(self, series: pd.Series, column: str, rules: Dict) -> List[ValidationResult]:
        """Validate categorical column"""
        results = []
        
        # Check allowed values
        if 'allowed_values' in rules:
            invalid_values = ~series.isin(rules['allowed_values'])
            invalid_count = invalid_values.sum()
            
            if invalid_count > 0:
                unique_invalid = series[invalid_values].unique()
                results.append(ValidationResult(
                    check_name=f"invalid_categorical_values_{column}",
                    status="fail",
                    message=f"'{column}' contains {invalid_count} invalid values: {list(unique_invalid)[:5]}",
                    affected_rows=invalid_count,
                    severity="high",
                    recommendations=[f"Ensure '{column}' values are from: {rules['allowed_values']}"]
                ))
        
        # Check for inconsistent naming (case, spacing)
        if column in ['product', 'brand']:
            inconsistencies = self._detect_naming_inconsistencies(series)
            results.extend(inconsistencies)
        
        return results
    
    def _validate_datetime_column(self, series: pd.Series, column: str, rules: Dict) -> List[ValidationResult]:
        """Validate datetime column"""
        results = []
        
        try:
            # Try to convert to datetime
            dt_series = pd.to_datetime(series, errors='coerce')
            invalid_dates = dt_series.isnull().sum()
            
            if invalid_dates > 0:
                results.append(ValidationResult(
                    check_name=f"invalid_datetime_{column}",
                    status="warning",
                    message=f"'{column}' has {invalid_dates} invalid datetime values",
                    affected_rows=invalid_dates,
                    severity="medium",
                    recommendations=[f"Fix invalid datetime format in '{column}'"]
                ))
            
            # Date range validation
            valid_dates = dt_series.dropna()
            
            if 'min_date' in rules and rules['min_date']:
                min_date = pd.to_datetime(rules['min_date'])
                too_old = (valid_dates < min_date).sum()
                if too_old > 0:
                    results.append(ValidationResult(
                        check_name=f"dates_too_old_{column}",
                        status="warning",
                        message=f"'{column}' has {too_old} dates before {rules['min_date']}",
                        affected_rows=too_old,
                        severity="low",
                        recommendations=["Check if very old dates are accurate"]
                    ))
            
            # Future dates check
            future_dates = (valid_dates > datetime.now()).sum()
            if future_dates > 0:
                results.append(ValidationResult(
                    check_name=f"future_dates_{column}",
                    status="warning",
                    message=f"'{column}' has {future_dates} future dates",
                    affected_rows=future_dates,
                    severity="medium",
                    recommendations=["Review future dates for accuracy"]
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name=f"datetime_validation_error_{column}",
                status="fail",
                message=f"Error validating datetime column '{column}': {str(e)}",
                affected_rows=len(series),
                severity="high",
                recommendations=[f"Check datetime format in '{column}' column"]
            ))
        
        return results
    
    def _validate_consistency(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate data consistency across columns"""
        results = []
        
        # Check product-brand consistency
        if 'product' in df.columns and 'brand' in df.columns:
            consistency_check = self._check_product_brand_consistency(df)
            results.extend(consistency_check)
        
        # Check rating-sentiment consistency
        if 'rating' in df.columns and 'sentiment_label' in df.columns:
            rating_sentiment_check = self._check_rating_sentiment_consistency(df)
            results.extend(rating_sentiment_check)
        
        return results
    
    def _validate_duplicates(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate duplicate records"""
        results = []
        
        # Exact duplicates
        exact_duplicates = df.duplicated().sum()
        if exact_duplicates > 0:
            results.append(ValidationResult(
                check_name="exact_duplicates",
                status="warning",
                message=f"Found {exact_duplicates} exact duplicate records",
                affected_rows=exact_duplicates,
                severity="medium",
                recommendations=["Remove exact duplicate records"]
            ))
        
        # Near-duplicate reviews (same product, similar text)
        if 'product' in df.columns and 'review_text' in df.columns:
            near_duplicates = self._detect_near_duplicates(df)
            if near_duplicates > 0:
                results.append(ValidationResult(
                    check_name="near_duplicate_reviews",
                    status="warning",
                    message=f"Found approximately {near_duplicates} near-duplicate reviews",
                    affected_rows=near_duplicates,
                    severity="low",
                    recommendations=["Review potential duplicate content for the same products"]
                ))
        
        return results
    
    def _validate_statistics(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate statistical properties of the data"""
        results = []
        
        # Rating distribution validation
        if 'rating' in df.columns:
            rating_stats = self._validate_rating_distribution(df['rating'])
            results.extend(rating_stats)
        
        # Review length distribution
        if 'review_text' in df.columns:
            length_stats = self._validate_text_length_distribution(df['review_text'])
            results.extend(length_stats)
        
        return results
    
    def _validate_business_rules(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate business-specific rules"""
        results = []
        
        # Check for unrealistic rating patterns
        if 'product' in df.columns and 'rating' in df.columns:
            rating_patterns = self._check_rating_patterns(df)
            results.extend(rating_patterns)
        
        # Check review authenticity indicators
        if 'review_text' in df.columns:
            authenticity_check = self._check_review_authenticity(df)
            results.extend(authenticity_check)
        
        return results
    
    # Helper methods for specific validations
    def _detect_suspicious_text_patterns(self, text_series: pd.Series) -> List[ValidationResult]:
        """Detect suspicious patterns in text"""
        results = []
        
        # Very repetitive text
        repetitive_count = 0
        for text in text_series.sample(min(100, len(text_series))):  # Sample for performance
            words = str(text).split()
            if len(words) > 5:
                word_counts = Counter(words)
                most_common_freq = word_counts.most_common(1)[0][1] if word_counts else 0
                if most_common_freq > len(words) * 0.5:  # More than 50% same word
                    repetitive_count += 1
        
        if repetitive_count > 0:
            results.append(ValidationResult(
                check_name="repetitive_text",
                status="warning",
                message=f"Found approximately {repetitive_count} reviews with highly repetitive text",
                affected_rows=repetitive_count,
                severity="low",
                recommendations=["Review text quality and consider filtering repetitive content"]
            ))
        
        return results
    
    def _detect_naming_inconsistencies(self, series: pd.Series) -> List[ValidationResult]:
        """Detect naming inconsistencies in categorical data"""
        results = []
        
        # Group similar names (case-insensitive, whitespace variations)
        name_groups = defaultdict(list)
        for name in series.unique():
            normalized = str(name).lower().strip().replace(' ', '')
            name_groups[normalized].append(name)
        
        inconsistencies = 0
        for normalized, names in name_groups.items():
            if len(names) > 1:
                inconsistencies += len(names) - 1
        
        if inconsistencies > 0:
            results.append(ValidationResult(
                check_name=f"naming_inconsistencies_{series.name}",
                status="warning",
                message=f"Found {inconsistencies} naming inconsistencies (case, spacing variations)",
                affected_rows=inconsistencies,
                severity="low",
                recommendations=[f"Standardize naming conventions in '{series.name}' column"]
            ))
        
        return results
    
    def _check_product_brand_consistency(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check if product-brand combinations are consistent"""
        results = []
        
        try:
            # Group by product and check if brand is consistent
            product_brands = df.groupby('product')['brand'].nunique()
            inconsistent_products = (product_brands > 1).sum()
            
            if inconsistent_products > 0:
                results.append(ValidationResult(
                    check_name="product_brand_inconsistency",
                    status="warning",
                    message=f"{inconsistent_products} products have inconsistent brand assignments",
                    affected_rows=inconsistent_products,
                    severity="medium",
                    recommendations=["Review and standardize product-brand relationships"]
                ))
        
        except Exception as e:
            logger.warning(f"Could not check product-brand consistency: {e}")
        
        return results
    
    def _check_rating_sentiment_consistency(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check consistency between ratings and sentiment"""
        results = []
        
        try:
            inconsistencies = 0
            
            # High ratings with negative sentiment
            high_rating_negative = ((df['rating'] >= 4) & (df['sentiment_label'] == 'negative')).sum()
            
            # Low ratings with positive sentiment  
            low_rating_positive = ((df['rating'] <= 2) & (df['sentiment_label'] == 'positive')).sum()
            
            inconsistencies = high_rating_negative + low_rating_positive
            
            if inconsistencies > 0:
                results.append(ValidationResult(
                    check_name="rating_sentiment_inconsistency",
                    status="warning",
                    message=f"Found {inconsistencies} records with inconsistent rating-sentiment combinations",
                    affected_rows=inconsistencies,
                    severity="medium",
                    recommendations=["Review sentiment analysis quality or rating accuracy"]
                ))
        
        except Exception as e:
            logger.warning(f"Could not check rating-sentiment consistency: {e}")
        
        return results
    
    def _detect_near_duplicates(self, df: pd.DataFrame) -> int:
        """Detect near-duplicate reviews (simplified implementation)"""
        try:
            if len(df) < 100:  # Skip for small datasets
                return 0
            
            # Sample for performance
            sample_df = df.sample(min(200, len(df)))
            near_duplicates = 0
            
            # Check for reviews with same product and very similar length
            for product in sample_df['product'].unique()[:10]:  # Limit products for performance
                product_reviews = sample_df[sample_df['product'] == product]['review_text']
                lengths = product_reviews.str.len()
                
                # Count reviews with very similar lengths (likely duplicates)
                for length in lengths:
                    similar_length = abs(lengths - length) <= 10  # Within 10 characters
                    if similar_length.sum() > 1:
                        near_duplicates += similar_length.sum() - 1
                        break
            
            return near_duplicates
        
        except Exception:
            return 0
    
    def _validate_rating_distribution(self, rating_series: pd.Series) -> List[ValidationResult]:
        """Validate rating distribution for anomalies"""
        results = []
        
        try:
            numeric_ratings = pd.to_numeric(rating_series, errors='coerce').dropna()
            
            if len(numeric_ratings) == 0:
                return results
            
            # Check for extremely skewed distributions
            rating_counts = numeric_ratings.value_counts()
            total_ratings = len(numeric_ratings)
            
            # Check if one rating dominates (>70% of all ratings)
            max_rating_pct = rating_counts.max() / total_ratings
            
            if max_rating_pct > 0.7:
                dominant_rating = rating_counts.idxmax()
                results.append(ValidationResult(
                    check_name="skewed_rating_distribution",
                    status="warning",
                    message=f"Rating distribution is highly skewed: {max_rating_pct:.1%} are {dominant_rating}-star ratings",
                    affected_rows=int(rating_counts.max()),
                    severity="medium",
                    recommendations=["Review rating collection process for potential bias"]
                ))
        
        except Exception as e:
            logger.warning(f"Could not validate rating distribution: {e}")
        
        return results
    
    def _validate_text_length_distribution(self, text_series: pd.Series) -> List[ValidationResult]:
        """Validate text length distribution"""
        results = []
        
        try:
            lengths = text_series.astype(str).str.len()
            
            # Check for suspiciously uniform lengths
            length_std = lengths.std()
            length_mean = lengths.mean()
            
            if length_std < length_mean * 0.1:  # Very low variance
                results.append(ValidationResult(
                    check_name="uniform_text_lengths",
                    status="warning",
                    message=f"Review text lengths are suspiciously uniform (std: {length_std:.1f}, mean: {length_mean:.1f})",
                    affected_rows=len(text_series),
                    severity="low",
                    recommendations=["Check if reviews are being truncated or templated"]
                ))
        
        except Exception as e:
            logger.warning(f"Could not validate text length distribution: {e}")
        
        return results
    
    def _check_rating_patterns(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check for unrealistic rating patterns"""
        results = []
        
        try:
            # Check for products with only extreme ratings (1s and 5s)
            for product in df['product'].unique()[:20]:  # Limit for performance
                product_ratings = df[df['product'] == product]['rating']
                numeric_ratings = pd.to_numeric(product_ratings, errors='coerce').dropna()
                
                if len(numeric_ratings) >= 10:  # Need sufficient data
                    extreme_ratings = ((numeric_ratings == 1) | (numeric_ratings == 5)).sum()
                    extreme_pct = extreme_ratings / len(numeric_ratings)
                    
                    if extreme_pct > 0.8:  # More than 80% extreme ratings
                        results.append(ValidationResult(
                            check_name="extreme_rating_pattern",
                            status="warning",
                            message=f"Product '{product}' has {extreme_pct:.1%} extreme ratings (1 or 5 stars)",
                            affected_rows=extreme_ratings,
                            severity="low",
                            recommendations=["Investigate potential rating manipulation or polarizing product"]
                        ))
        
        except Exception as e:
            logger.warning(f"Could not check rating patterns: {e}")
        
        return results
    
    def _check_review_authenticity(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check for indicators of inauthentic reviews"""
        results = []
        
        try:
            if 'review_text' not in df.columns:
                return results
            
            # Check for very short reviews that are overly positive
            if 'sentiment_label' in df.columns:
                short_positive = df[
                    (df['review_text'].str.len() < 20) & 
                    (df['sentiment_label'] == 'positive')
                ]
                
                if len(short_positive) > len(df) * 0.3:  # More than 30% short positive
                    results.append(ValidationResult(
                        check_name="suspicious_short_positive",
                        status="warning",
                        message=f"High percentage ({len(short_positive)/len(df):.1%}) of very short positive reviews",
                        affected_rows=len(short_positive),
                        severity="medium",
                        recommendations=["Review the authenticity of very short positive reviews"]
                    ))
        
        except Exception as e:
            logger.warning(f"Could not check review authenticity: {e}")
        
        return results
    
    def _calculate_quality_metrics(self, df: pd.DataFrame, validation_results: List[ValidationResult]) -> QualityMetrics:
        """Calculate overall data quality metrics"""
        
        # Count issues by severity
        critical_issues = len([r for r in validation_results if r.severity == 'critical'])
        high_issues = len([r for r in validation_results if r.severity == 'high'])
        medium_issues = len([r for r in validation_results if r.severity == 'medium'])
        low_issues = len([r for r in validation_results if r.severity == 'low'])
        
        warnings = len([r for r in validation_results if r.status == 'warning'])
        failures = len([r for r in validation_results if r.status == 'fail'])
        
        # Calculate component scores
        completeness_score = self._calculate_completeness_score(df)
        validity_score = self._calculate_validity_score(validation_results)
        consistency_score = self._calculate_consistency_score(validation_results)
        uniqueness_score = self._calculate_uniqueness_score(df, validation_results)
        
        # Calculate overall score (weighted average)
        overall_score = (
            completeness_score * 0.3 +
            validity_score * 0.3 +
            consistency_score * 0.2 +
            uniqueness_score * 0.2
        )
        
        # Penalize for critical issues
        if critical_issues > 0:
            overall_score = min(overall_score, 40.0)
        elif high_issues > 0:
            overall_score = min(overall_score, 70.0)
        
        return QualityMetrics(
            overall_score=overall_score,
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=consistency_score,
            uniqueness_score=uniqueness_score,
            total_records=len(df),
            issues_found=len(validation_results),
            critical_issues=critical_issues + high_issues,
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def _calculate_completeness_score(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score"""
        if df.empty:
            return 0.0
        
        essential_columns = ['product', 'review_text']
        available_essential = [col for col in essential_columns if col in df.columns]
        
        if not available_essential:
            return 0.0
        
        # Calculate fill rate for essential columns
        total_cells = len(df) * len(available_essential)
        filled_cells = 0
        
        for col in available_essential:
            filled_cells += df[col].notna().sum()
        
        completeness = (filled_cells / total_cells) * 100 if total_cells > 0 else 0
        return min(completeness, 100.0)
    
    def _calculate_validity_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate data validity score based on validation results"""
        if not validation_results:
            return 100.0
        
        # Start with perfect score and deduct for issues
        validity_score = 100.0
        
        for result in validation_results:
            if result.status == 'fail':
                if result.severity == 'critical':
                    validity_score -= 20
                elif result.severity == 'high':
                    validity_score -= 10
                elif result.severity == 'medium':
                    validity_score -= 5
            elif result.status == 'warning':
                if result.severity == 'high':
                    validity_score -= 5
                elif result.severity == 'medium':
                    validity_score -= 2
                else:
                    validity_score -= 1
        
        return max(validity_score, 0.0)
    
    def _calculate_consistency_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate data consistency score"""
        consistency_issues = [
            r for r in validation_results 
            if 'consistency' in r.check_name or 'inconsistency' in r.check_name
        ]
        
        if not consistency_issues:
            return 100.0
        
        consistency_score = 100.0
        for issue in consistency_issues:
            if issue.severity == 'high':
                consistency_score -= 15
            elif issue.severity == 'medium':
                consistency_score -= 8
            else:
                consistency_score -= 3
        
        return max(consistency_score, 0.0)
    
    def _calculate_uniqueness_score(self, df: pd.DataFrame, validation_results: List[ValidationResult]) -> float:
        """Calculate data uniqueness score"""
        duplicate_issues = [
            r for r in validation_results 
            if 'duplicate' in r.check_name
        ]
        
        if not duplicate_issues:
            return 100.0
        
        uniqueness_score = 100.0
        for issue in duplicate_issues:
            duplicate_rate = issue.affected_rows / len(df) if len(df) > 0 else 0
            uniqueness_score -= duplicate_rate * 50  # Penalize based on duplicate rate
        
        return max(uniqueness_score, 0.0)
    
    def _create_empty_metrics(self) -> QualityMetrics:
        """Create empty quality metrics for error cases"""
        return QualityMetrics(
            overall_score=0.0,
            completeness_score=0.0,
            validity_score=0.0,
            consistency_score=0.0,
            uniqueness_score=0.0,
            total_records=0,
            issues_found=0,
            critical_issues=0,
            warnings=0,
            timestamp=datetime.now()
        )
    
    def get_quality_report(self, validation_results: List[ValidationResult], quality_metrics: QualityMetrics) -> str:
        """Generate a human-readable quality report"""
        
        # Quality level determination
        score = quality_metrics.overall_score
        if score >= self.quality_thresholds['excellent']:
            quality_level = "Excellent ✅"
            quality_color = "🟢"
        elif score >= self.quality_thresholds['good']:
            quality_level = "Good ✅"
            quality_color = "🟢"
        elif score >= self.quality_thresholds['fair']:
            quality_level = "Fair ⚠️"
            quality_color = "🟡"
        elif score >= self.quality_thresholds['poor']:
            quality_level = "Poor ⚠️"
            quality_color = "🟠"
        else:
            quality_level = "Critical ❌"
            quality_color = "🔴"
        
        report = f"""
📊 DATA QUALITY REPORT
{'='*50}

{quality_color} OVERALL QUALITY SCORE: {score:.1f}/100 ({quality_level})

📈 COMPONENT SCORES:
   • Completeness: {quality_metrics.completeness_score:.1f}/100
   • Validity: {quality_metrics.validity_score:.1f}/100  
   • Consistency: {quality_metrics.consistency_score:.1f}/100
   • Uniqueness: {quality_metrics.uniqueness_score:.1f}/100

📋 SUMMARY:
   • Total Records: {quality_metrics.total_records:,}
   • Issues Found: {quality_metrics.issues_found}
   • Critical Issues: {quality_metrics.critical_issues}
   • Warnings: {quality_metrics.warnings}

"""
        
        # Add top issues
        if validation_results:
            critical_high = [r for r in validation_results if r.severity in ['critical', 'high']]
            if critical_high:
                report += "🚨 CRITICAL & HIGH PRIORITY ISSUES:\n"
                for i, result in enumerate(critical_high[:5], 1):
                    severity_icon = "🔴" if result.severity == 'critical' else "🟠"
                    report += f"   {i}. {severity_icon} {result.message}\n"
                    if result.recommendations:
                        report += f"      → {result.recommendations[0]}\n"
                report += "\n"
        
        # Add recommendations
        report += "💡 RECOMMENDED ACTIONS:\n"
        
        if quality_metrics.critical_issues > 0:
            report += "   • Address critical issues immediately\n"
        
        if quality_metrics.completeness_score < 90:
            report += "   • Improve data completeness by filling missing values\n"
        
        if quality_metrics.validity_score < 85:
            report += "   • Fix data validation errors and inconsistencies\n"
        
        if quality_metrics.uniqueness_score < 95:
            report += "   • Remove or consolidate duplicate records\n"
        
        if quality_metrics.overall_score >= 85:
            report += "   • Continue monitoring data quality over time\n"
        
        report += f"\n📅 Report Generated: {quality_metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "="*50
        
        return report
    
    def get_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        if df is None or df.empty:
            return {'error': 'Empty or invalid dataset'}
        
        profile = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'column_names': list(df.columns)
            },
            'column_profiles': {}
        }
        
        # Profile each column
        for column in df.columns:
            col_profile = self._profile_column(df[column])
            profile['column_profiles'][column] = col_profile
        
        return profile
    
    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile a single column"""
        profile = {
            'data_type': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100
        }
        
        # Type-specific profiling
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if not numeric_series.empty:
                profile.update({
                    'min_value': numeric_series.min(),
                    'max_value': numeric_series.max(),
                    'mean': numeric_series.mean(),
                    'median': numeric_series.median(),
                    'std_dev': numeric_series.std(),
                    'quartiles': {
                        'Q1': numeric_series.quantile(0.25),
                        'Q2': numeric_series.quantile(0.5),
                        'Q3': numeric_series.quantile(0.75)
                    }
                })
        
        elif pd.api.types.is_object_dtype(series):
            text_series = series.astype(str)
            profile.update({
                'avg_length': text_series.str.len().mean(),
                'min_length': text_series.str.len().min(),
                'max_length': text_series.str.len().max(),
                'most_common': series.value_counts().head(3).to_dict()
            })
        
        return profile

# Usage example
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'product': ['iPhone 15'] * 50 + ['Galaxy S24'] * 45 + [None] * 5,
        'rating': [4.5] * 30 + [3.8] * 40 + [6.0] * 10 + [None] * 20,  # Include invalid ratings
        'review_text': ['Great phone with excellent camera'] * 80 + [''] * 10 + [None] * 10,
        'brand': ['Apple'] * 50 + ['Samsung'] * 45 + ['samsung'] * 5,  # Include inconsistency
        'sentiment_label': ['positive'] * 70 + ['negative'] * 20 + ['invalid'] * 10
    })
    
    # Initialize validator
    validator = DataQualityValidator()
    
    # Validate dataset
    results, metrics = validator.validate_dataset(sample_data)
    
    # Print report
    report = validator.get_quality_report(results, metrics)
    print(report)
    
    # Print data profile
    profile = validator.get_data_profile(sample_data)
    print("\n📊 DATA PROFILE:")
    print(f"Rows: {profile['basic_info']['total_rows']}")
    print(f"Columns: {profile['basic_info']['total_columns']}")
    print(f"Memory: {profile['basic_info']['memory_usage']}")