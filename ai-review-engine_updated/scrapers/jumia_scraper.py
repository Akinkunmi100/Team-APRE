"""
Jumia E-commerce Platform Scraper
"""

from typing import List, Dict, Optional
from datetime import datetime
import logging
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class JumiaScraper(BaseScraper):
    """Scraper for Jumia e-commerce platform"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.jumia.com.ng"
        
    def scrape_reviews(self, product_url: str) -> List[Dict]:
        """
        Scrape reviews from Jumia product page
        
        Args:
            product_url: URL of the product page
            
        Returns:
            List of review dictionaries
        """
        reviews = []
        
        # Get page content
        html = self.get_page_content(product_url, use_selenium=True)
        if not html:
            logger.error(f"Failed to fetch page: {product_url}")
            return reviews
        
        soup = self.parse_html(html)
        
        # Find review containers (adjust selectors based on actual Jumia structure)
        review_elements = soup.find_all('article', class_='review')
        
        for review_elem in review_elements:
            try:
                review = self._extract_review(review_elem)
                if review:
                    reviews.append(review)
            except Exception as e:
                logger.error(f"Error extracting review: {e}")
                continue
        
        logger.info(f"Scraped {len(reviews)} reviews from {product_url}")
        return reviews
    
    def _extract_review(self, review_elem) -> Optional[Dict]:
        """Extract review data from review element"""
        review = {}
        
        try:
            # Extract reviewer name
            name_elem = review_elem.find('div', class_='reviewer-name')
            review['reviewer_name'] = self.clean_text(name_elem.text) if name_elem else "Anonymous"
            
            # Extract rating
            rating_elem = review_elem.find('div', class_='stars')
            if rating_elem:
                rating_text = rating_elem.get('data-rating', '')
                review['rating'] = self.extract_rating(rating_text)
            else:
                review['rating'] = None
            
            # Extract review title
            title_elem = review_elem.find('h3', class_='review-title')
            review['title'] = self.clean_text(title_elem.text) if title_elem else ""
            
            # Extract review text
            text_elem = review_elem.find('p', class_='review-text')
            review['text'] = self.clean_text(text_elem.text) if text_elem else ""
            
            # Extract date
            date_elem = review_elem.find('span', class_='review-date')
            if date_elem:
                review['date'] = self._parse_date(date_elem.text)
            else:
                review['date'] = datetime.now().isoformat()
            
            # Extract verified purchase flag
            verified_elem = review_elem.find('span', class_='verified-purchase')
            review['verified_purchase'] = bool(verified_elem)
            
            # Add metadata
            review['source'] = 'Jumia'
            review['scraped_at'] = datetime.now().isoformat()
            
            return review
            
        except Exception as e:
            logger.error(f"Error in _extract_review: {e}")
            return None
    
    def scrape_product_info(self, product_url: str) -> Dict:
        """
        Scrape product information from Jumia
        
        Args:
            product_url: URL of the product page
            
        Returns:
            Dictionary containing product information
        """
        product_info = {}
        
        # Get page content
        html = self.get_page_content(product_url, use_selenium=True)
        if not html:
            logger.error(f"Failed to fetch product page: {product_url}")
            return product_info
        
        soup = self.parse_html(html)
        
        try:
            # Extract product name
            name_elem = soup.find('h1', class_='product-title')
            product_info['name'] = self.clean_text(name_elem.text) if name_elem else ""
            
            # Extract brand
            brand_elem = soup.find('a', class_='product-brand')
            product_info['brand'] = self.clean_text(brand_elem.text) if brand_elem else ""
            
            # Extract price
            price_elem = soup.find('span', class_='price')
            if price_elem:
                product_info['price'] = self._extract_price(price_elem.text)
            
            # Extract overall rating
            rating_elem = soup.find('div', class_='product-rating')
            if rating_elem:
                rating_text = rating_elem.find('span', class_='rating-value')
                product_info['overall_rating'] = self.extract_rating(rating_text.text) if rating_text else None
            
            # Extract total reviews count
            review_count_elem = soup.find('span', class_='review-count')
            if review_count_elem:
                product_info['total_reviews'] = self._extract_number(review_count_elem.text)
            
            # Extract product specifications
            specs = {}
            spec_table = soup.find('div', class_='product-specs')
            if spec_table:
                spec_rows = spec_table.find_all('div', class_='spec-row')
                for row in spec_rows:
                    key_elem = row.find('span', class_='spec-key')
                    value_elem = row.find('span', class_='spec-value')
                    if key_elem and value_elem:
                        key = self.clean_text(key_elem.text).lower().replace(' ', '_')
                        specs[key] = self.clean_text(value_elem.text)
            
            product_info['specifications'] = specs
            product_info['url'] = product_url
            product_info['source'] = 'Jumia'
            product_info['scraped_at'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error scraping product info: {e}")
        
        return product_info
    
    def _parse_date(self, date_text: str) -> str:
        """Parse date from text"""
        # Implement date parsing logic based on Jumia's date format
        # For now, return current date
        return datetime.now().isoformat()
    
    def _extract_price(self, price_text: str) -> Optional[float]:
        """Extract numerical price from text"""
        try:
            import re
            # Remove currency symbols and extract number
            price_text = re.sub(r'[₦$,]', '', price_text)
            match = re.search(r'(\d+\.?\d*)', price_text)
            if match:
                return float(match.group(1))
        except:
            pass
        return None
    
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract integer from text"""
        try:
            import re
            match = re.search(r'(\d+)', text)
            if match:
                return int(match.group(1))
        except:
            pass
        return None
    
    def search_phones(self, query: str = "phones", max_pages: int = 5) -> List[str]:
        """
        Search for phones on Jumia and return product URLs
        
        Args:
            query: Search query
            max_pages: Maximum number of search result pages to scrape
            
        Returns:
            List of product URLs
        """
        product_urls = []
        
        for page in range(1, max_pages + 1):
            search_url = f"{self.base_url}/catalog/?q={query}&page={page}"
            html = self.get_page_content(search_url)
            
            if not html:
                continue
            
            soup = self.parse_html(html)
            
            # Find product links
            product_links = soup.find_all('a', class_='product-link')
            for link in product_links:
                href = link.get('href')
                if href:
                    full_url = f"{self.base_url}{href}" if not href.startswith('http') else href
                    product_urls.append(full_url)
        
        logger.info(f"Found {len(product_urls)} products for query: {query}")
        return product_urls
