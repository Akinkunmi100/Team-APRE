"""
Database Manager Wrapper
"""

from database.models import db_manager

# Export the database manager instance
DatabaseManager = db_manager.__class__
