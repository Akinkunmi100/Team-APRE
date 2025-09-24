"""
Data Quality Validator for AI Phone Review Engine
Implements content validation, source reliability scoring, and data freshness checks
"""

import re
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import statistics
from collections import Counter, defaultdict
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Enumeration of data source types"""
    SCRAPED_REVIEW = "scraped_review"
    API_RESULT = "api_result"
    PRICING_DATA = "pricing_data"
    SPECIFICATION = "specification"
    USER_REVIEW = "user_review"
    EXPERT_REVIEW = "expert_review"

class QualityLevel(Enum):
    """Quality levels for data scoring"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 70-89%
    FAIR = "fair"           # 50-69%
    POOR = "poor"           # 30-49%
    UNACCEPTABLE = "unacceptable"  # 0-29%

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_score: float
    quality_level: QualityLevel
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

@dataclass
class SourceReliabilityScore:
    """Reliability scoring for data sources"""
    source_name: str
    base_reliability: float
    historical_accuracy: float
    content_quality: float
    update_frequency: float
    overall_score: float
    last_updated: str

@dataclass
class DataFreshnessCheck:
    """Data freshness validation"""
    data_age_hours: float
    is_fresh: bool
    freshness_score: float
    recommended_refresh: bool
    staleness_reason: Optional[str]

class ContentValidator:
    """Validates content quality and authenticity"""
    
    def __init__(self):
        # Spam/fake content patterns
        self.spam_patterns = [
            r'\b(?:click here|buy now|limited time|act now|hurry)\b',
            r'\b(?:free|guaranteed|amazing|incredible|unbelievable)\b.*!{2,}',
            r'(?:http[s]?://|www\.)\S+',  # URLs in reviews
            r'\b(?:viagra|casino|poker|lottery|winner)\b',
            r'[A-Z]{3,}\s[A-Z]{3,}',  # ALL CAPS spam
            r'(.)\1{4,}',  # Repeated characters
            r'^\s*[!@#$%^&*()_+=\[\]{}|;:",.<>?/~`-]{5,}\s*$'  # Symbol spam
        ]
        
        # Quality indicators
        self.quality_indicators = {
            'specific_details': r'\b(?:battery|camera|display|processor|ram|storage|price|performance)\b',
            'technical_terms': r'\b(?:mah|ghz|mp|gb|tb|fps|ppi|oled|amoled|lcd)\b',
            'comparative_language': r'\b(?:better|worse|compared|versus|vs|than|similar)\b',
            'experiential_language': r'\b(?:used|tried|tested|experience|noticed|found)\b'
        }
        
        # Profanity filter (basic)
        self.profanity_words = {
            'mild': ['damn', 'hell', 'crap'],
            'moderate': ['stupid', 'dumb', 'sucks'],
            'severe': []  # Would include stronger terms
        }
    
    def validate_review_content(self, content: str, source: str) -> ValidationResult:
        """Validate review content quality"""
        
        issues = []
        warnings = []
        quality_scores = []
        
        # Basic validation
        if not content or len(content.strip()) < 10:
            issues.append("Content too short or empty")
            return ValidationResult(False, 0.0, QualityLevel.UNACCEPTABLE, issues, warnings, {})
        
        # Length quality scoring
        content_length = len(content)
        if content_length < 50:
            quality_scores.append(0.3)
            warnings.append("Very short content")
        elif content_length < 150:
            quality_scores.append(0.6)
            warnings.append("Short content")
        elif content_length > 2000:
            quality_scores.append(0.8)
            warnings.append("Very long content - may contain irrelevant information")
        else:
            quality_scores.append(0.9)
        
        # Spam detection
        spam_score = self._detect_spam(content)
        if spam_score > 0.7:
            issues.append(f"High spam probability: {spam_score:.2f}")
        elif spam_score > 0.4:
            warnings.append(f"Moderate spam indicators: {spam_score:.2f}")
        
        quality_scores.append(1 - spam_score)
        
        # Content quality assessment
        quality_score = self._assess_content_quality(content)
        quality_scores.append(quality_score)
        
        # Language quality
        language_score = self._assess_language_quality(content)
        quality_scores.append(language_score)
        
        # Profanity check
        profanity_score = self._check_profanity(content)
        if profanity_score > 0.5:
            warnings.append(f"Contains inappropriate language: {profanity_score:.2f}")
        quality_scores.append(1 - profanity_score)
        
        # Calculate overall quality
        overall_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        quality_level = self._get_quality_level(overall_quality)
        
        is_valid = len(issues) == 0 and overall_quality >= 0.3
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=overall_quality,
            quality_level=quality_level,
            issues=issues,
            warnings=warnings,
            metadata={
                'content_length': content_length,
                'spam_score': spam_score,
                'language_score': language_score,
                'profanity_score': profanity_score,
                'source': source,
                'validation_timestamp': datetime.now().isoformat()
            }
        )
    
    def validate_specification_data(self, specs: Dict[str, Any], source: str) -> ValidationResult:
        """Validate specification data completeness and accuracy"""
        
        issues = []
        warnings = []
        quality_scores = []
        
        if not specs:
            issues.append("No specification data provided")
            return ValidationResult(False, 0.0, QualityLevel.UNACCEPTABLE, issues, warnings, {})
        
        # Essential specifications check
        essential_specs = [
            'display', 'processor', 'ram', 'storage', 'camera', 'battery', 'os'
        ]
        
        present_specs = 0
        for spec in essential_specs:
            if any(spec.lower() in key.lower() for key in specs.keys()):
                present_specs += 1
        
        completeness_score = present_specs / len(essential_specs)
        quality_scores.append(completeness_score)
        
        if completeness_score < 0.3:
            issues.append(f"Missing essential specifications: {present_specs}/{len(essential_specs)}")
        elif completeness_score < 0.6:
            warnings.append(f"Some essential specifications missing: {present_specs}/{len(essential_specs)}")
        
        # Data format validation
        format_score = self._validate_spec_formats(specs)
        quality_scores.append(format_score)
        
        # Value consistency check
        consistency_score = self._check_spec_consistency(specs)
        quality_scores.append(consistency_score)
        
        overall_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        quality_level = self._get_quality_level(overall_quality)
        
        is_valid = len(issues) == 0 and overall_quality >= 0.4
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=overall_quality,
            quality_level=quality_level,
            issues=issues,
            warnings=warnings,
            metadata={
                'spec_count': len(specs),
                'completeness_score': completeness_score,
                'format_score': format_score,
                'consistency_score': consistency_score,
                'source': source
            }
        )
    
    def validate_pricing_data(self, pricing_data: List[Dict], source: str) -> ValidationResult:
        """Validate pricing data accuracy and reasonableness"""
        
        issues = []
        warnings = []
        quality_scores = []
        
        if not pricing_data:
            issues.append("No pricing data provided")
            return ValidationResult(False, 0.0, QualityLevel.UNACCEPTABLE, issues, warnings, {})
        
        # Price range validation
        prices = []
        for item in pricing_data:
            if 'price' in item:
                try:
                    price = float(item['price'])
                    if price <= 0:
                        issues.append(f"Invalid price: {price}")
                    elif price > 10000:  # Unreasonably high for phones
                        warnings.append(f"Unusually high price: ${price}")
                    elif price < 50:  # Unreasonably low for phones
                        warnings.append(f"Unusually low price: ${price}")
                    else:
                        prices.append(price)
                except ValueError:
                    issues.append(f"Non-numeric price: {item['price']}")
        
        if not prices:
            issues.append("No valid prices found")
            return ValidationResult(False, 0.0, QualityLevel.UNACCEPTABLE, issues, warnings, {})
        
        # Price consistency check
        if len(prices) > 1:
            price_std = statistics.stdev(prices)
            price_mean = statistics.mean(prices)
            coefficient_of_variation = price_std / price_mean if price_mean > 0 else 0
            
            if coefficient_of_variation > 0.5:
                warnings.append(f"High price variation: {coefficient_of_variation:.2f}")
                quality_scores.append(0.6)
            else:
                quality_scores.append(0.9)
        else:
            quality_scores.append(0.7)  # Single price gets moderate score
        
        # Data freshness check
        current_time = datetime.now()
        fresh_data_count = 0
        
        for item in pricing_data:
            if 'last_updated' in item:
                try:
                    update_time = datetime.fromisoformat(item['last_updated'].replace('Z', '+00:00'))
                    age_hours = (current_time - update_time).total_seconds() / 3600
                    
                    if age_hours <= 24:
                        fresh_data_count += 1
                    elif age_hours > 168:  # 1 week
                        warnings.append(f"Stale pricing data: {age_hours:.1f} hours old")
                except ValueError:
                    warnings.append("Invalid timestamp format")
        
        freshness_score = fresh_data_count / len(pricing_data) if pricing_data else 0
        quality_scores.append(freshness_score)
        
        overall_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        quality_level = self._get_quality_level(overall_quality)
        
        is_valid = len(issues) == 0 and overall_quality >= 0.4
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=overall_quality,
            quality_level=quality_level,
            issues=issues,
            warnings=warnings,
            metadata={
                'price_count': len(prices),
                'price_range': {'min': min(prices), 'max': max(prices)} if prices else {},
                'freshness_score': freshness_score,
                'source': source
            }
        )
    
    def _detect_spam(self, content: str) -> float:
        """Detect spam content using pattern matching"""
        
        spam_score = 0.0
        content_lower = content.lower()
        
        # Check spam patterns
        pattern_matches = 0
        for pattern in self.spam_patterns:
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            pattern_matches += matches
        
        # Normalize by content length
        if len(content) > 0:
            spam_score += min(pattern_matches / (len(content) / 100), 0.8)
        
        # Check for excessive punctuation
        punct_count = len(re.findall(r'[!?]{2,}', content))
        spam_score += min(punct_count * 0.1, 0.3)
        
        # Check for excessive capitalization
        if len(content) > 10:
            caps_ratio = len(re.findall(r'[A-Z]', content)) / len(content)
            if caps_ratio > 0.3:
                spam_score += min((caps_ratio - 0.3) * 2, 0.4)
        
        return min(spam_score, 1.0)
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality based on indicators"""
        
        quality_score = 0.5  # Base score
        content_lower = content.lower()
        
        # Check for quality indicators
        for indicator_type, pattern in self.quality_indicators.items():
            matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
            if matches > 0:
                quality_score += min(matches * 0.05, 0.15)
        
        # Sentence structure quality
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(valid_sentences) >= 2:
            quality_score += 0.1
        
        # Word diversity
        words = re.findall(r'\b\w+\b', content_lower)
        if words:
            unique_words = set(words)
            diversity_ratio = len(unique_words) / len(words)
            quality_score += min(diversity_ratio, 0.2)
        
        return min(quality_score, 1.0)
    
    def _assess_language_quality(self, content: str) -> float:
        """Assess language quality and readability"""
        
        # Basic grammar checks
        grammar_score = 0.7  # Start with decent score
        
        # Check for basic sentence structure
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(valid_sentences) == 0:
            return 0.2
        
        # Average sentence length check
        avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
        
        if 5 <= avg_sentence_length <= 20:
            grammar_score += 0.1
        elif avg_sentence_length > 30:
            grammar_score -= 0.1
        
        # Check for proper capitalization
        capitalized_sentences = sum(1 for s in valid_sentences if s.strip() and s.strip()[0].isupper())
        if len(valid_sentences) > 0:
            capitalization_ratio = capitalized_sentences / len(valid_sentences)
            if capitalization_ratio > 0.7:
                grammar_score += 0.1
        
        # Check for excessive repetition
        words = re.findall(r'\b\w+\b', content.lower())
        if words:
            word_counts = Counter(words)
            max_repetition = max(word_counts.values())
            repetition_ratio = max_repetition / len(words)
            
            if repetition_ratio > 0.2:
                grammar_score -= min(repetition_ratio - 0.2, 0.3)
        
        return max(min(grammar_score, 1.0), 0.0)
    
    def _check_profanity(self, content: str) -> float:
        """Check for inappropriate language"""
        
        profanity_score = 0.0
        content_lower = content.lower()
        
        # Check different levels of profanity
        for level, words in self.profanity_words.items():
            for word in words:
                if word in content_lower:
                    if level == 'mild':
                        profanity_score += 0.1
                    elif level == 'moderate':
                        profanity_score += 0.3
                    elif level == 'severe':
                        profanity_score += 0.7
        
        return min(profanity_score, 1.0)
    
    def _validate_spec_formats(self, specs: Dict[str, Any]) -> float:
        """Validate specification data formats"""
        
        format_score = 0.8  # Start with good score
        total_specs = len(specs)
        
        if total_specs == 0:
            return 0.0
        
        format_issues = 0
        
        # Check common specification formats
        for key, value in specs.items():
            key_lower = key.lower()
            value_str = str(value).lower()
            
            # Display format check
            if 'display' in key_lower or 'screen' in key_lower:
                if not re.search(r'\d+\.?\d*\s*inch', value_str):
                    format_issues += 1
            
            # RAM format check
            elif 'ram' in key_lower or 'memory' in key_lower:
                if not re.search(r'\d+\s*gb', value_str):
                    format_issues += 1
            
            # Storage format check
            elif 'storage' in key_lower:
                if not re.search(r'\d+\s*(gb|tb)', value_str):
                    format_issues += 1
            
            # Battery format check
            elif 'battery' in key_lower:
                if not re.search(r'\d+\s*mah', value_str):
                    format_issues += 1
        
        if format_issues > 0:
            format_score -= (format_issues / total_specs) * 0.4
        
        return max(format_score, 0.2)
    
    def _check_spec_consistency(self, specs: Dict[str, Any]) -> float:
        """Check specification consistency and logical validity"""
        
        consistency_score = 0.9
        
        # Extract numeric values for consistency checking
        ram_gb = self._extract_numeric_spec(specs, ['ram', 'memory'])
        storage_gb = self._extract_numeric_spec(specs, ['storage'])
        display_inch = self._extract_numeric_spec(specs, ['display', 'screen'])
        battery_mah = self._extract_numeric_spec(specs, ['battery'])
        
        # Logical consistency checks
        if ram_gb and ram_gb > 32:  # Unlikely for phones currently
            consistency_score -= 0.1
        
        if storage_gb and storage_gb > 2048:  # 2TB+ unlikely
            consistency_score -= 0.1
        
        if display_inch and (display_inch < 3 or display_inch > 8):  # Reasonable phone screen sizes
            consistency_score -= 0.2
        
        if battery_mah and (battery_mah < 1000 or battery_mah > 10000):  # Reasonable battery sizes
            consistency_score -= 0.1
        
        return max(consistency_score, 0.3)
    
    def _extract_numeric_spec(self, specs: Dict[str, Any], keywords: List[str]) -> Optional[float]:
        """Extract numeric value from specifications"""
        
        for key, value in specs.items():
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in keywords):
                # Extract first number from the value
                match = re.search(r'(\d+\.?\d*)', str(value))
                if match:
                    return float(match.group(1))
        
        return None
    
    def _get_quality_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level"""
        
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.FAIR
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE

class SourceReliabilityTracker:
    """Tracks and scores source reliability over time"""
    
    def __init__(self, config_file: str = "config/source_reliability.json"):
        self.config_file = config_file
        self.reliability_data = {}
        self.load_reliability_data()
    
    def load_reliability_data(self):
        """Load historical reliability data"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.reliability_data = json.load(f)
            else:
                # Initialize with default reliability scores
                self.reliability_data = {
                    'gsmarena': {'base_reliability': 0.9, 'historical_accuracy': 0.85, 'content_quality': 0.9},
                    'phonearena': {'base_reliability': 0.8, 'historical_accuracy': 0.8, 'content_quality': 0.85},
                    'cnet': {'base_reliability': 0.85, 'historical_accuracy': 0.8, 'content_quality': 0.9},
                    'techradar': {'base_reliability': 0.8, 'historical_accuracy': 0.75, 'content_quality': 0.8},
                    'ebay': {'base_reliability': 0.6, 'historical_accuracy': 0.7, 'content_quality': 0.6},
                    'bestbuy': {'base_reliability': 0.85, 'historical_accuracy': 0.9, 'content_quality': 0.8},
                    'google_shopping': {'base_reliability': 0.75, 'historical_accuracy': 0.8, 'content_quality': 0.7}
                }
                self.save_reliability_data()
        except Exception as e:
            logger.error(f"Error loading reliability data: {e}")
            self.reliability_data = {}
    
    def save_reliability_data(self):
        """Save reliability data to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.reliability_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving reliability data: {e}")
    
    def get_source_reliability(self, source_name: str) -> SourceReliabilityScore:
        """Get reliability score for a source"""
        
        source_data = self.reliability_data.get(source_name, {
            'base_reliability': 0.5,
            'historical_accuracy': 0.5,
            'content_quality': 0.5
        })
        
        # Calculate update frequency score (how recently we've seen data from this source)
        update_frequency = self._calculate_update_frequency(source_name)
        
        # Overall score is weighted average
        overall_score = (
            source_data['base_reliability'] * 0.4 +
            source_data['historical_accuracy'] * 0.3 +
            source_data['content_quality'] * 0.2 +
            update_frequency * 0.1
        )
        
        return SourceReliabilityScore(
            source_name=source_name,
            base_reliability=source_data['base_reliability'],
            historical_accuracy=source_data['historical_accuracy'],
            content_quality=source_data['content_quality'],
            update_frequency=update_frequency,
            overall_score=overall_score,
            last_updated=datetime.now().isoformat()
        )
    
    def update_source_reliability(self, source_name: str, validation_result: ValidationResult):
        """Update source reliability based on validation results"""
        
        if source_name not in self.reliability_data:
            self.reliability_data[source_name] = {
                'base_reliability': 0.5,
                'historical_accuracy': 0.5,
                'content_quality': 0.5,
                'validation_history': []
            }
        
        # Add validation result to history
        if 'validation_history' not in self.reliability_data[source_name]:
            self.reliability_data[source_name]['validation_history'] = []
        
        self.reliability_data[source_name]['validation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'quality_score': validation_result.quality_score,
            'is_valid': validation_result.is_valid,
            'issues_count': len(validation_result.issues)
        })
        
        # Keep only recent history (last 100 validations)
        history = self.reliability_data[source_name]['validation_history'][-100:]
        self.reliability_data[source_name]['validation_history'] = history
        
        # Update scores based on recent performance
        if len(history) >= 5:
            recent_scores = [h['quality_score'] for h in history[-20:]]  # Last 20 validations
            avg_quality = statistics.mean(recent_scores)
            
            # Gradually adjust content quality score
            current_quality = self.reliability_data[source_name]['content_quality']
            self.reliability_data[source_name]['content_quality'] = (
                current_quality * 0.8 + avg_quality * 0.2
            )
            
            # Update historical accuracy
            valid_count = sum(1 for h in history[-20:] if h['is_valid'])
            accuracy = valid_count / len(history[-20:])
            
            current_accuracy = self.reliability_data[source_name]['historical_accuracy']
            self.reliability_data[source_name]['historical_accuracy'] = (
                current_accuracy * 0.8 + accuracy * 0.2
            )
        
        self.save_reliability_data()
    
    def _calculate_update_frequency(self, source_name: str) -> float:
        """Calculate how frequently we receive updates from a source"""
        
        source_data = self.reliability_data.get(source_name, {})
        history = source_data.get('validation_history', [])
        
        if len(history) < 2:
            return 0.5  # Default score for new sources
        
        # Calculate average time between updates
        timestamps = [datetime.fromisoformat(h['timestamp']) for h in history[-10:]]
        if len(timestamps) < 2:
            return 0.5
        
        time_diffs = [
            (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Convert to hours
            for i in range(1, len(timestamps))
        ]
        
        avg_hours_between_updates = statistics.mean(time_diffs)
        
        # Score based on update frequency
        if avg_hours_between_updates <= 1:
            return 1.0  # Very frequent updates
        elif avg_hours_between_updates <= 6:
            return 0.8  # Frequent updates
        elif avg_hours_between_updates <= 24:
            return 0.6  # Daily updates
        elif avg_hours_between_updates <= 168:
            return 0.4  # Weekly updates
        else:
            return 0.2  # Infrequent updates

class DataFreshnessChecker:
    """Checks data freshness and recommends updates"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'pricing_max_age_hours': 6,      # Pricing data stales quickly
            'review_max_age_hours': 168,     # Reviews valid for a week
            'spec_max_age_hours': 720,       # Specs valid for a month
            'news_max_age_hours': 24         # News stales quickly
        }
    
    def check_data_freshness(self, data_timestamp: str, data_type: str) -> DataFreshnessCheck:
        """Check if data is fresh enough for use"""
        
        try:
            data_time = datetime.fromisoformat(data_timestamp.replace('Z', '+00:00'))
            current_time = datetime.now()
            age_hours = (current_time - data_time).total_seconds() / 3600
            
            max_age_key = f'{data_type.lower()}_max_age_hours'
            max_age = self.config.get(max_age_key, 168)  # Default to 1 week
            
            is_fresh = age_hours <= max_age
            freshness_score = max(0.0, 1.0 - (age_hours / max_age))
            
            # Determine if refresh is recommended
            recommend_refresh = age_hours > (max_age * 0.8)
            
            staleness_reason = None
            if not is_fresh:
                if age_hours > max_age * 2:
                    staleness_reason = "Very stale data"
                else:
                    staleness_reason = "Moderately stale data"
            
            return DataFreshnessCheck(
                data_age_hours=age_hours,
                is_fresh=is_fresh,
                freshness_score=freshness_score,
                recommended_refresh=recommend_refresh,
                staleness_reason=staleness_reason
            )
            
        except ValueError:
            logger.error(f"Invalid timestamp format: {data_timestamp}")
            return DataFreshnessCheck(
                data_age_hours=float('inf'),
                is_fresh=False,
                freshness_score=0.0,
                recommended_refresh=True,
                staleness_reason="Invalid timestamp"
            )

# Main validator class
class DataQualityValidator:
    """Main data quality validation orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'min_quality_threshold': 0.5,
            'enable_source_tracking': True,
            'enable_freshness_checks': True
        }
        
        self.content_validator = ContentValidator()
        self.reliability_tracker = SourceReliabilityTracker()
        self.freshness_checker = DataFreshnessChecker()
    
    async def validate_data(self, data: Dict[str, Any], data_type: DataSourceType, 
                          source: str) -> ValidationResult:
        """Main validation method for any type of data"""
        
        try:
            # Route to appropriate validator
            if data_type == DataSourceType.SCRAPED_REVIEW:
                result = self.content_validator.validate_review_content(
                    data.get('content', ''), source
                )
            elif data_type == DataSourceType.SPECIFICATION:
                result = self.content_validator.validate_specification_data(
                    data.get('specifications', {}), source
                )
            elif data_type == DataSourceType.PRICING_DATA:
                result = self.content_validator.validate_pricing_data(
                    data.get('pricing', []), source
                )
            else:
                # Generic validation for other types
                result = self._validate_generic_data(data, source)
            
            # Update source reliability if enabled
            if self.config['enable_source_tracking']:
                self.reliability_tracker.update_source_reliability(source, result)
            
            # Add freshness check if timestamp is available
            if self.config['enable_freshness_checks'] and 'timestamp' in data:
                freshness = self.freshness_checker.check_data_freshness(
                    data['timestamp'], data_type.value
                )
                result.metadata['freshness'] = asdict(freshness)
            
            # Add source reliability info
            if self.config['enable_source_tracking']:
                reliability = self.reliability_tracker.get_source_reliability(source)
                result.metadata['source_reliability'] = asdict(reliability)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for {source}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                quality_level=QualityLevel.UNACCEPTABLE,
                issues=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={'error': str(e), 'source': source}
            )
    
    def _validate_generic_data(self, data: Dict[str, Any], source: str) -> ValidationResult:
        """Generic validation for unknown data types"""
        
        issues = []
        warnings = []
        
        if not data:
            issues.append("Empty data")
            return ValidationResult(False, 0.0, QualityLevel.UNACCEPTABLE, issues, warnings, {})
        
        # Basic completeness check
        completeness_score = len([v for v in data.values() if v is not None]) / len(data)
        
        if completeness_score < 0.5:
            warnings.append(f"Low data completeness: {completeness_score:.2f}")
        
        return ValidationResult(
            is_valid=completeness_score >= 0.3,
            quality_score=completeness_score,
            quality_level=self.content_validator._get_quality_level(completeness_score),
            issues=issues,
            warnings=warnings,
            metadata={'completeness_score': completeness_score, 'source': source}
        )
    
    def get_source_reliability_summary(self) -> Dict[str, SourceReliabilityScore]:
        """Get reliability summary for all tracked sources"""
        
        summary = {}
        for source_name in self.reliability_tracker.reliability_data.keys():
            summary[source_name] = self.reliability_tracker.get_source_reliability(source_name)
        
        return summary

# Factory functions
def create_data_quality_validator(config=None):
    """Create configured data quality validator"""
    return DataQualityValidator(config=config)