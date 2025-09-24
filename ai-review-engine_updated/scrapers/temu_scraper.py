"""
Temu E-commerce Platform Scraper
"""

from typing import List, Dict, Optional
from datetime import datetime
import logging
import json
import time
from .base_scraper import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)


class TemuScraper(BaseScraper):
    """Scraper for Temu e-commerce platform"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.temu.com"
        
    def scrape_reviews(self, product_url: str) -> List[Dict]:
        """
        Scrape reviews from Temu product page
        
        Args:
            product_url: URL of the product page
            
        Returns:
            List of review dictionaries
        """
        reviews = []
        
        # Temu requires Selenium due to heavy JavaScript
        html = self.get_page_content(product_url, use_selenium=True)
        if not html:
            logger.error(f"Failed to fetch page: {product_url}")
            return reviews
        
        soup = self.parse_html(html)
        
        # Find review containers (Temu-specific selectors)
        review_containers = soup.find_all('div', {'class': lambda x: x and 'review-item' in x})
        
        for container in review_containers:
            try:
                review = self._extract_review_temu(container)
                if review:
                    reviews.append(review)
            except Exception as e:
                logger.error(f"Error extracting Temu review: {e}")
                continue
        
        logger.info(f"Scraped {len(reviews)} reviews from Temu")
        return reviews
    
    def _extract_review_temu(self, review_elem) -> Optional[Dict]:
        """Extract review data from Temu review element"""
        review = {}
        
        try:
            # Extract reviewer info
            reviewer_elem = review_elem.find('div', class_='reviewer-info')
            if reviewer_elem:
                review['reviewer_name'] = self.clean_text(reviewer_elem.text)
            else:
                review['reviewer_name'] = "Anonymous"
            
            # Extract rating (Temu uses star system)
            stars = review_elem.find_all('svg', class_='star-icon')
            filled_stars = len([s for s in stars if 'filled' in s.get('class', [])])
            review['rating'] = filled_stars
            
            # Extract review text
            text_elem = review_elem.find('div', class_='review-content')
            review['text'] = self.clean_text(text_elem.text) if text_elem else ""
            
            # Extract images if present
            images = review_elem.find_all('img', class_='review-image')
            review['has_images'] = len(images) > 0
            review['image_count'] = len(images)
            
            # Extract date
            date_elem = review_elem.find('span', class_='review-date')
            if date_elem:
                review['date'] = self._parse_temu_date(date_elem.text)
            else:
                review['date'] = datetime.now().isoformat()
            
            # Extract helpful votes
            helpful_elem = review_elem.find('span', class_='helpful-count')
            if helpful_elem:
                review['helpful_count'] = self._extract_number(helpful_elem.text)
            else:
                review['helpful_count'] = 0
            
            # Temu-specific: Extract product variation
            variation_elem = review_elem.find('span', class_='product-variation')
            if variation_elem:
                review['product_variation'] = self.clean_text(variation_elem.text)
            
            # Add metadata
            review['source'] = 'Temu'
            review['scraped_at'] = datetime.now().isoformat()
            review['platform_specific'] = {
                'has_media': review['has_images'],
                'is_verified': self._check_verified_purchase_temu(review_elem)
            }
            
            return review
            
        except Exception as e:
            logger.error(f"Error in _extract_review_temu: {e}")
            return None
    
    def _check_verified_purchase_temu(self, review_elem) -> bool:
        """Check if review is from verified purchase on Temu"""
        verified_badge = review_elem.find('span', class_='verified-badge')
        return verified_badge is not None
    
    def _parse_temu_date(self, date_text: str) -> str:
        """Parse Temu-specific date format"""
        # Temu often uses relative dates like "2 days ago"
        import re
        from datetime import timedelta
        
        date_text = date_text.lower()
        
        if 'ago' in date_text:
            # Extract number and unit
            match = re.search(r'(\d+)\s*(day|week|month|hour)', date_text)
            if match:
                num = int(match.group(1))
                unit = match.group(2)
                
                if unit == 'hour':
                    delta = timedelta(hours=num)
                elif unit == 'day':
                    delta = timedelta(days=num)
                elif unit == 'week':
                    delta = timedelta(weeks=num)
                elif unit == 'month':
                    delta = timedelta(days=num * 30)  # Approximate
                else:
                    delta = timedelta()
                
                return (datetime.now() - delta).isoformat()
        
        return datetime.now().isoformat()
    
    def scrape_product_info(self, product_url: str) -> Dict:
        """
        Scrape product information from Temu
        
        Args:
            product_url: URL of the product page
            
        Returns:
            Dictionary containing product information
        """
        product_info = {}
        
        html = self.get_page_content(product_url, use_selenium=True)
        if not html:
            return product_info
        
        soup = self.parse_html(html)
        
        try:
            # Product name
            name_elem = soup.find('h1', class_='product-title')
            product_info['name'] = self.clean_text(name_elem.text) if name_elem else ""
            
            # Price (Temu often shows discounted prices)
            price_elem = soup.find('span', class_='current-price')
            if price_elem:
                product_info['current_price'] = self._extract_price(price_elem.text)
            
            original_price_elem = soup.find('span', class_='original-price')
            if original_price_elem:
                product_info['original_price'] = self._extract_price(original_price_elem.text)
                
            # Calculate discount percentage
            if 'current_price' in product_info and 'original_price' in product_info:
                if product_info['original_price'] > 0:
                    discount = ((product_info['original_price'] - product_info['current_price']) 
                               / product_info['original_price']) * 100
                    product_info['discount_percentage'] = round(discount, 2)
            
            # Rating
            rating_elem = soup.find('span', class_='product-rating')
            if rating_elem:
                product_info['overall_rating'] = self.extract_rating(rating_elem.text)
            
            # Number of reviews
            review_count_elem = soup.find('span', class_='review-count')
            if review_count_elem:
                product_info['total_reviews'] = self._extract_number(review_count_elem.text)
            
            # Number of orders (Temu shows this)
            orders_elem = soup.find('span', class_='order-count')
            if orders_elem:
                product_info['total_orders'] = self._extract_number(orders_elem.text)
            
            # Shipping info
            shipping_elem = soup.find('div', class_='shipping-info')
            if shipping_elem:
                product_info['shipping_info'] = self.clean_text(shipping_elem.text)
            
            # Product specifications
            specs = {}
            spec_list = soup.find('div', class_='product-specs')
            if spec_list:
                spec_items = spec_list.find_all('div', class_='spec-item')
                for item in spec_items:
                    label = item.find('span', class_='spec-label')
                    value = item.find('span', class_='spec-value')
                    if label and value:
                        specs[self.clean_text(label.text)] = self.clean_text(value.text)
            
            product_info['specifications'] = specs
            product_info['url'] = product_url
            product_info['source'] = 'Temu'
            product_info['scraped_at'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error scraping Temu product info: {e}")
        
        return product_info
    
    def search_products(self, query: str, category: str = "cell-phones",
                       max_products: int = 20) -> List[str]:
        """
        Search for products on Temu
        
        Args:
            query: Search query
            category: Product category
            max_products: Maximum number of products to return
            
        Returns:
            List of product URLs
        """
        product_urls = []
        search_url = f"{self.base_url}/search?q={query}&category={category}"
        
        html = self.get_page_content(search_url, use_selenium=True)
        if not html:
            return product_urls
        
        soup = self.parse_html(html)
        
        # Find product cards
        product_cards = soup.find_all('div', class_='product-card')[:max_products]
        
        for card in product_cards:
            link = card.find('a', class_='product-link')
            if link and link.get('href'):
                full_url = f"{self.base_url}{link['href']}" if not link['href'].startswith('http') else link['href']
                product_urls.append(full_url)
        
        logger.info(f"Found {len(product_urls)} products on Temu for query: {query}")
        return product_urls
    
    def _extract_number(self, text: str) -> int:
        """Extract number from text, handling K/M abbreviations"""
        import re
        
        text = text.upper()
        
        # Handle K (thousands) and M (millions)
        if 'K' in text:
            match = re.search(r'([\d.]+)K', text)
            if match:
                return int(float(match.group(1)) * 1000)
        elif 'M' in text:
            match = re.search(r'([\d.]+)M', text)
            if match:
                return int(float(match.group(1)) * 1000000)
        
        # Regular number extraction
        match = re.search(r'(\d+)', text.replace(',', ''))
        if match:
            return int(match.group(1))
        
        return 0
