"""
Base Scraper Class for Web Scraping
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all web scrapers"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the base scraper with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.headers = {
            'User-Agent': self.config['scraping']['user_agent']
        }
        self.timeout = self.config['scraping']['timeout']
        self.max_retries = self.config['scraping']['max_retries']
        self.delay = self.config['scraping']['delay_between_requests']
        
    def get_page_content(self, url: str, use_selenium: bool = False) -> Optional[str]:
        """
        Fetch page content using requests or Selenium
        
        Args:
            url: URL to scrape
            use_selenium: Whether to use Selenium for JavaScript-rendered pages
            
        Returns:
            HTML content of the page
        """
        if use_selenium:
            return self._get_with_selenium(url)
        else:
            return self._get_with_requests(url)
    
    def _get_with_requests(self, url: str) -> Optional[str]:
        """Fetch page using requests library"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                time.sleep(self.delay)  # Rate limiting
                return response.text
            except requests.RequestException as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * 2)  # Back off on retry
        return None
    
    def _get_with_selenium(self, url: str) -> Optional[str]:
        """Fetch page using Selenium for JavaScript-rendered content"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(f'user-agent={self.headers["User-Agent"]}')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            time.sleep(self.delay)
            return driver.page_source
            
        except Exception as e:
            logger.error(f"Selenium error fetching {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content using BeautifulSoup"""
        return BeautifulSoup(html, 'html.parser')
    
    @abstractmethod
    def scrape_reviews(self, product_url: str) -> List[Dict]:
        """
        Abstract method to scrape reviews from a product page
        
        Args:
            product_url: URL of the product page
            
        Returns:
            List of review dictionaries
        """
        pass
    
    @abstractmethod
    def scrape_product_info(self, product_url: str) -> Dict:
        """
        Abstract method to scrape product information
        
        Args:
            product_url: URL of the product page
            
        Returns:
            Dictionary containing product information
        """
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep punctuation
        text = text.strip()
        return text
    
    def extract_rating(self, rating_text: str) -> Optional[float]:
        """Extract numerical rating from text"""
        try:
            # Try to extract number from string like "4.5 stars" or "4.5/5"
            import re
            match = re.search(r'(\d+\.?\d*)', rating_text)
            if match:
                return float(match.group(1))
        except:
            pass
        return None
