#!/usr/bin/env python
"""
Setup script for AI Phone Review Engine
Handles installation, configuration, and initialization
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
import argparse
import platform
from typing import Dict, List, Optional


class SetupManager:
    """Manages the setup process for the AI Phone Review Engine"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.config_dir = self.root_dir / "config"
        self.cache_dir = self.root_dir / "cache"
        self.logs_dir = self.root_dir / "logs"
        self.data_dir = self.root_dir / "data"
        self.platform = platform.system()
        
    def run(self, profile: str = "minimal"):
        """
        Run the complete setup process
        
        Args:
            profile: Installation profile (minimal, standard, full)
        """
        print("=" * 60)
        print("AI Phone Review Engine - Setup")
        print("=" * 60)
        
        # Check Python version
        self.check_python_version()
        
        # Create directory structure
        self.create_directories()
        
        # Generate configuration files
        self.generate_configs()
        
        # Install dependencies
        self.install_dependencies(profile)
        
        # Download required models/data
        self.download_resources(profile)
        
        # Initialize database
        self.initialize_database()
        
        # Run verification tests
        self.verify_installation()
        
        print("\n✅ Setup completed successfully!")
        print("\nTo start the application, run:")
        print("  streamlit run app.py")
        print("\nFor the complete system with all features, run:")
        print("  streamlit run main_engine.py")
    
    def check_python_version(self):
        """Check if Python version meets requirements"""
        print("\n📌 Checking Python version...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
            sys.exit(1)
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    
    def create_directories(self):
        """Create necessary directory structure"""
        print("\n📁 Creating directory structure...")
        
        directories = [
            self.config_dir,
            self.cache_dir,
            self.cache_dir / "models",
            self.logs_dir,
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.root_dir / "tests",
            self.root_dir / "static",
            self.root_dir / "uploads"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {directory.relative_to(self.root_dir)}")
    
    def generate_configs(self):
        """Generate default configuration files"""
        print("\n⚙️ Generating configuration files...")
        
        # Main configuration
        main_config = {
            'app': {
                'name': 'AI Phone Review Engine',
                'version': '2.0.0',
                'debug': False,
                'host': '0.0.0.0',
                'port': 8501
            },
            'database': {
                'url': 'sqlite:///data/review_engine.db',
                'pool_size': 10,
                'max_overflow': 20
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'enabled': False
            },
            'api': {
                'base_url': 'http://localhost:8000',
                'timeout': 30,
                'rate_limit': 100
            },
            'models': {
                'cache_dir': 'cache/models',
                'download_on_start': False,
                'use_gpu': False
            },
            'sentiment_analysis': {
                'aspects': ['camera', 'battery', 'screen', 'performance', 'price', 'design', 'software'],
                'sentiment_labels': ['positive', 'negative', 'neutral'],
                'confidence_threshold': 0.7
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/app.log',
                'max_bytes': 10485760,
                'backup_count': 5
            },
            'security': {
                'secret_key': self._generate_secret_key(),
                'algorithm': 'HS256',
                'token_expire_minutes': 30
            }
        }
        
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(main_config, f, default_flow_style=False)
        print(f"  ✓ Generated {config_file.name}")
        
        # Environment configuration
        env_config = f"""
# AI Phone Review Engine Environment Configuration

# Application
APP_NAME="AI Phone Review Engine"
DEBUG=False
SECRET_KEY="{self._generate_secret_key()}"

# Database
DATABASE_URL="sqlite:///data/review_engine.db"
# For PostgreSQL: postgresql://user:password@localhost/dbname

# Redis (optional)
REDIS_HOST="localhost"
REDIS_PORT=6379
REDIS_ENABLED=False

# API
API_BASE_URL="http://localhost:8000"
API_KEY=""

# Models
USE_GPU=False
MODEL_CACHE_DIR="cache/models"

# Logging
LOG_LEVEL="INFO"
LOG_FILE="logs/app.log"
"""
        
        env_file = self.root_dir / ".env"
        with open(env_file, 'w') as f:
            f.write(env_config.strip())
        print(f"  ✓ Generated {env_file.name}")
        
        # Create .env.example
        example_file = self.root_dir / ".env.example"
        with open(example_file, 'w') as f:
            f.write(env_config.replace(self._generate_secret_key(), "your-secret-key-here"))
        print(f"  ✓ Generated {example_file.name}")
    
    def install_dependencies(self, profile: str):
        """
        Install Python dependencies based on profile
        
        Args:
            profile: Installation profile (minimal, standard, full)
        """
        print(f"\n📦 Installing dependencies ({profile} profile)...")
        
        requirements_map = {
            'minimal': 'requirements-minimal.txt',
            'standard': 'requirements.txt',
            'full': 'requirements.txt'
        }
        
        requirements_file = requirements_map.get(profile, 'requirements-minimal.txt')
        
        try:
            # Upgrade pip first
            print("  Upgrading pip...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True, text=True)
            
            # Install requirements
            print(f"  Installing from {requirements_file}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                print(f"⚠️ Some packages failed to install:")
                print(result.stderr)
                print("\nContinuing with available packages...")
            else:
                print("  ✅ All dependencies installed successfully")
                
            # Install additional packages for full profile
            if profile == 'full':
                optional_packages = [
                    'transformers',
                    'torch',
                    'tensorflow',
                    'spacy',
                    'en_core_web_sm'
                ]
                
                print("\n  Installing optional ML packages...")
                for package in optional_packages:
                    try:
                        if package == 'en_core_web_sm':
                            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'],
                                         capture_output=True, text=True)
                        else:
                            subprocess.run([sys.executable, '-m', 'pip', 'install', package],
                                         capture_output=True, text=True)
                        print(f"    ✓ {package}")
                    except:
                        print(f"    ⚠️ {package} (optional)")
                        
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            print("Please install manually using: pip install -r requirements-minimal.txt")
    
    def download_resources(self, profile: str):
        """Download required models and resources"""
        print("\n📥 Downloading resources...")
        
        if profile == 'minimal':
            print("  Skipping model downloads for minimal profile")
            return
        
        # Create sample data if not exists
        sample_data_file = self.data_dir / "sample_data.csv"
        if not sample_data_file.exists():
            self._create_sample_data(sample_data_file)
            print(f"  ✓ Created sample data")
        
        # Download NLTK data
        try:
            import nltk
            nltk_data = ['punkt', 'vader_lexicon', 'stopwords']
            for data in nltk_data:
                try:
                    nltk.download(data, quiet=True)
                    print(f"  ✓ NLTK {data}")
                except:
                    print(f"  ⚠️ NLTK {data} (optional)")
        except ImportError:
            print("  ⚠️ NLTK not available")
    
    def initialize_database(self):
        """Initialize the database"""
        print("\n🗄️ Initializing database...")
        
        try:
            from database.database_manager import DatabaseManager
            db = DatabaseManager()
            db.create_tables()
            print("  ✅ Database initialized")
        except Exception as e:
            print(f"  ⚠️ Database initialization failed: {e}")
            print("  You can initialize it later manually")
    
    def verify_installation(self):
        """Verify the installation by running basic tests"""
        print("\n🔍 Verifying installation...")
        
        # Check core imports
        required_modules = [
            'streamlit',
            'pandas',
            'numpy',
            'plotly',
            'fastapi',
            'sqlalchemy'
        ]
        
        failed_imports = []
        for module in required_modules:
            try:
                __import__(module)
                print(f"  ✓ {module}")
            except ImportError:
                failed_imports.append(module)
                print(f"  ❌ {module}")
        
        if failed_imports:
            print(f"\n⚠️ Some core modules are missing: {', '.join(failed_imports)}")
            print("Please install them manually")
        else:
            print("\n✅ All core modules verified")
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def _create_sample_data(self, file_path: Path):
        """Create sample CSV data for testing"""
        import pandas as pd
        from datetime import datetime, timedelta
        import random
        
        # Generate sample reviews
        products = ['iPhone 15 Pro', 'Samsung S24', 'Google Pixel 8', 'OnePlus 12']
        sentiments = ['positive', 'negative', 'neutral']
        
        data = []
        for i in range(100):
            data.append({
                'review_id': f'REV{i:04d}',
                'product': random.choice(products),
                'rating': random.randint(1, 5),
                'review_text': self._generate_sample_review(),
                'sentiment': random.choice(sentiments),
                'date': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
                'verified_purchase': random.choice([True, False]),
                'helpful_votes': random.randint(0, 100)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def _generate_sample_review(self) -> str:
        """Generate a sample review text"""
        templates = [
            "Great phone with excellent {feature}. The {aspect} could be better though.",
            "Disappointed with the {feature}. Expected more from this price range.",
            "Amazing {aspect}! Best phone I've ever owned.",
            "Average performance. The {feature} is okay but nothing special.",
            "Terrible experience with {aspect}. Would not recommend."
        ]
        
        features = ['camera', 'battery life', 'screen', 'performance', 'build quality']
        aspects = ['display', 'software', 'design', 'value', 'features']
        
        import random
        template = random.choice(templates)
        return template.format(
            feature=random.choice(features),
            aspect=random.choice(aspects)
        )


def main():
    """Main entry point for setup script"""
    parser = argparse.ArgumentParser(description='Setup AI Phone Review Engine')
    parser.add_argument(
        '--profile',
        choices=['minimal', 'standard', 'full'],
        default='minimal',
        help='Installation profile (default: minimal)'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset configuration and cache'
    )
    
    args = parser.parse_args()
    
    setup = SetupManager()
    
    if args.reset:
        print("Resetting configuration and cache...")
        # Clear cache and configs
        import shutil
        if setup.cache_dir.exists():
            shutil.rmtree(setup.cache_dir)
        if setup.config_dir.exists():
            shutil.rmtree(setup.config_dir)
        print("Reset complete")
    
    setup.run(profile=args.profile)


if __name__ == '__main__':
    main()
