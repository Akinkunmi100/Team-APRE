"""
User Feedback and Crowdsourced Data System for AI Phone Review Engine
Enables community contributions, user reviews, and collaborative phone information
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import sqlite3
from collections import defaultdict
import statistics
import pickle
import threading
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContributionType(Enum):
    REVIEW = "review"
    SPECIFICATION = "specification"
    PRICE_UPDATE = "price_update"
    AVAILABILITY = "availability"
    CORRECTION = "correction"
    FEATURE_REQUEST = "feature_request"
    SOURCE_SUGGESTION = "source_suggestion"
    RATING = "rating"

class ContributionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"

class UserRole(Enum):
    CONTRIBUTOR = "contributor"
    MODERATOR = "moderator"
    EXPERT = "expert"
    ADMIN = "admin"

@dataclass
class UserContribution:
    """Structure for user contributions"""
    contribution_id: str
    user_id: str
    contribution_type: ContributionType
    phone_name: str
    data: Dict[str, Any]
    status: ContributionStatus
    submitted_at: str
    reviewed_at: Optional[str]
    reviewer_id: Optional[str]
    confidence_score: float
    votes_up: int = 0
    votes_down: int = 0
    comments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserProfile:
    """User profile for contributors"""
    user_id: str
    username: str
    email: str
    role: UserRole
    reputation_score: float
    total_contributions: int
    approved_contributions: int
    expertise_areas: List[str]
    joined_at: str
    last_active: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    achievements: List[str] = field(default_factory=list)

@dataclass
class CrowdsourcedPhoneData:
    """Aggregated crowdsourced phone information"""
    phone_name: str
    aggregated_rating: Optional[float]
    user_reviews: List[Dict[str, Any]]
    specifications: Dict[str, Any]
    price_data: Dict[str, Any]
    pros_cons: Dict[str, List[str]]
    usage_experiences: List[Dict[str, Any]]
    common_issues: List[str]
    recommended_alternatives: List[str]
    community_confidence: float
    last_updated: str
    contributor_count: int

@dataclass
class ModerationTask:
    """Moderation task for reviewing contributions"""
    task_id: str
    contribution_id: str
    priority: int  # 1-5, higher is more urgent
    assigned_moderator: Optional[str]
    created_at: str
    due_date: str
    notes: str
    completed: bool = False

class UserFeedbackSystem:
    """System for managing user feedback and crowdsourced data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize user feedback system"""
        
        self.config = config or {
            'database_file': 'data/user_feedback.db',
            'min_reputation_for_direct_approval': 85,
            'votes_required_for_approval': 3,
            'max_pending_contributions_per_user': 10,
            'enable_gamification': True,
            'enable_email_notifications': False,
            'contribution_expiry_days': 30,
            'auto_moderation_threshold': 0.8,
            'reputation_weights': {
                'approved_contribution': 10,
                'rejected_contribution': -5,
                'upvote_received': 2,
                'downvote_received': -1,
                'helpful_comment': 3
            }
        }
        
        # Initialize database
        self.db_path = Path(self.config['database_file'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Thread lock for database operations
        self.db_lock = threading.Lock()
        
        # In-memory caches
        self.user_cache = {}
        self.contribution_cache = {}
        self.phone_data_cache = {}
        
        # Load existing data
        self._load_caches()
    
    def _init_database(self):
        """Initialize SQLite database tables"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    reputation_score REAL DEFAULT 0,
                    total_contributions INTEGER DEFAULT 0,
                    approved_contributions INTEGER DEFAULT 0,
                    expertise_areas TEXT,
                    joined_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    preferences TEXT,
                    achievements TEXT
                )
            """)
            
            # Contributions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contributions (
                    contribution_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    contribution_type TEXT NOT NULL,
                    phone_name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewer_id TEXT,
                    confidence_score REAL NOT NULL,
                    votes_up INTEGER DEFAULT 0,
                    votes_down INTEGER DEFAULT 0,
                    comments TEXT,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Votes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS votes (
                    vote_id TEXT PRIMARY KEY,
                    contribution_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    vote_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (contribution_id) REFERENCES contributions (contribution_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    UNIQUE(contribution_id, user_id)
                )
            """)
            
            # Moderation tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS moderation_tasks (
                    task_id TEXT PRIMARY KEY,
                    contribution_id TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    assigned_moderator TEXT,
                    created_at TEXT NOT NULL,
                    due_date TEXT NOT NULL,
                    notes TEXT,
                    completed INTEGER DEFAULT 0,
                    FOREIGN KEY (contribution_id) REFERENCES contributions (contribution_id)
                )
            """)
            
            # Phone data aggregations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS phone_data_aggregations (
                    phone_name TEXT PRIMARY KEY,
                    aggregated_data TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    contributor_count INTEGER NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_contributions_phone ON contributions(phone_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_contributions_user ON contributions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_contributions_status ON contributions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_votes_contribution ON votes(contribution_id)")
            
            conn.commit()
    
    def register_user(self, username: str, email: str, expertise_areas: List[str] = None) -> UserProfile:
        """Register a new user"""
        
        user_id = str(uuid.uuid4())
        user = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            role=UserRole.CONTRIBUTOR,
            reputation_score=0.0,
            total_contributions=0,
            approved_contributions=0,
            expertise_areas=expertise_areas or [],
            joined_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat()
        )
        
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO users 
                    (user_id, username, email, role, reputation_score, total_contributions, 
                     approved_contributions, expertise_areas, joined_at, last_active, 
                     preferences, achievements)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.user_id, user.username, user.email, user.role.value,
                    user.reputation_score, user.total_contributions, user.approved_contributions,
                    json.dumps(user.expertise_areas), user.joined_at, user.last_active,
                    json.dumps(user.preferences), json.dumps(user.achievements)
                ))
                
                conn.commit()
        
        self.user_cache[user_id] = user
        logger.info(f"Registered new user: {username}")
        return user
    
    def submit_contribution(self, user_id: str, contribution_type: ContributionType, 
                          phone_name: str, data: Dict[str, Any], 
                          metadata: Dict[str, Any] = None) -> UserContribution:
        """Submit a user contribution"""
        
        # Check if user exists and has capacity
        user = self.get_user_profile(user_id)
        if not user:
            raise ValueError("User not found")
        
        pending_count = self._count_pending_contributions(user_id)
        if pending_count >= self.config['max_pending_contributions_per_user']:
            raise ValueError("Too many pending contributions")
        
        # Create contribution
        contribution_id = str(uuid.uuid4())
        confidence_score = self._calculate_contribution_confidence(user, data, contribution_type)
        
        contribution = UserContribution(
            contribution_id=contribution_id,
            user_id=user_id,
            contribution_type=contribution_type,
            phone_name=phone_name,
            data=data,
            status=ContributionStatus.PENDING,
            submitted_at=datetime.now().isoformat(),
            reviewed_at=None,
            reviewer_id=None,
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
        
        # Auto-approve high-reputation users with high confidence
        if (user.reputation_score >= self.config['min_reputation_for_direct_approval'] 
            and confidence_score >= self.config['auto_moderation_threshold']):
            contribution.status = ContributionStatus.APPROVED
            contribution.reviewed_at = datetime.now().isoformat()
            contribution.reviewer_id = "auto_moderator"
            self._update_user_reputation(user_id, 'approved_contribution')
        else:
            # Create moderation task
            self._create_moderation_task(contribution_id)
        
        # Save to database
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO contributions 
                    (contribution_id, user_id, contribution_type, phone_name, data, status,
                     submitted_at, reviewed_at, reviewer_id, confidence_score, votes_up,
                     votes_down, comments, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    contribution.contribution_id, contribution.user_id, contribution.contribution_type.value,
                    contribution.phone_name, json.dumps(contribution.data), contribution.status.value,
                    contribution.submitted_at, contribution.reviewed_at, contribution.reviewer_id,
                    contribution.confidence_score, contribution.votes_up, contribution.votes_down,
                    json.dumps(contribution.comments), json.dumps(contribution.metadata)
                ))
                
                # Update user statistics
                cursor.execute("""
                    UPDATE users SET total_contributions = total_contributions + 1,
                                   last_active = ?
                    WHERE user_id = ?
                """, (datetime.now().isoformat(), user_id))
                
                conn.commit()
        
        self.contribution_cache[contribution_id] = contribution
        
        # Update phone data if approved
        if contribution.status == ContributionStatus.APPROVED:
            self._update_phone_data_aggregation(phone_name)
        
        logger.info(f"Submitted contribution {contribution_id} by user {user_id}")
        return contribution
    
    def vote_on_contribution(self, user_id: str, contribution_id: str, vote_type: str) -> bool:
        """Vote on a contribution (upvote/downvote)"""
        
        if vote_type not in ['upvote', 'downvote']:
            raise ValueError("Invalid vote type")
        
        vote_id = str(uuid.uuid4())
        
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user already voted
                cursor.execute("""
                    SELECT vote_id FROM votes 
                    WHERE contribution_id = ? AND user_id = ?
                """, (contribution_id, user_id))
                
                if cursor.fetchone():
                    return False  # Already voted
                
                # Insert vote
                cursor.execute("""
                    INSERT INTO votes (vote_id, contribution_id, user_id, vote_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (vote_id, contribution_id, user_id, vote_type, datetime.now().isoformat()))
                
                # Update contribution vote counts
                if vote_type == 'upvote':
                    cursor.execute("""
                        UPDATE contributions SET votes_up = votes_up + 1 WHERE contribution_id = ?
                    """, (contribution_id,))
                else:
                    cursor.execute("""
                        UPDATE contributions SET votes_down = votes_down + 1 WHERE contribution_id = ?
                    """, (contribution_id,))
                
                conn.commit()
        
        # Update contributor reputation
        contribution = self.get_contribution(contribution_id)
        if contribution:
            reputation_change = 'upvote_received' if vote_type == 'upvote' else 'downvote_received'
            self._update_user_reputation(contribution.user_id, reputation_change)
        
        # Check if contribution should be auto-approved based on votes
        self._check_vote_based_approval(contribution_id)
        
        return True
    
    def moderate_contribution(self, moderator_id: str, contribution_id: str, 
                            decision: ContributionStatus, notes: str = "") -> bool:
        """Moderate a contribution"""
        
        # Verify moderator permissions
        moderator = self.get_user_profile(moderator_id)
        if not moderator or moderator.role not in [UserRole.MODERATOR, UserRole.EXPERT, UserRole.ADMIN]:
            return False
        
        contribution = self.get_contribution(contribution_id)
        if not contribution:
            return False
        
        # Update contribution status
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE contributions 
                    SET status = ?, reviewed_at = ?, reviewer_id = ?
                    WHERE contribution_id = ?
                """, (decision.value, datetime.now().isoformat(), moderator_id, contribution_id))
                
                # Update user statistics
                if decision == ContributionStatus.APPROVED:
                    cursor.execute("""
                        UPDATE users SET approved_contributions = approved_contributions + 1
                        WHERE user_id = ?
                    """, (contribution.user_id,))
                
                # Complete moderation task
                cursor.execute("""
                    UPDATE moderation_tasks SET completed = 1 WHERE contribution_id = ?
                """, (contribution_id,))
                
                conn.commit()
        
        # Update reputation
        reputation_change = 'approved_contribution' if decision == ContributionStatus.APPROVED else 'rejected_contribution'
        self._update_user_reputation(contribution.user_id, reputation_change)
        
        # Update phone data if approved
        if decision == ContributionStatus.APPROVED:
            self._update_phone_data_aggregation(contribution.phone_name)
        
        logger.info(f"Moderated contribution {contribution_id}: {decision.value}")
        return True
    
    def get_crowdsourced_phone_data(self, phone_name: str) -> Optional[CrowdsourcedPhoneData]:
        """Get aggregated crowdsourced data for a phone"""
        
        # Check cache first
        if phone_name in self.phone_data_cache:
            return self.phone_data_cache[phone_name]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT aggregated_data, last_updated, contributor_count
                FROM phone_data_aggregations
                WHERE phone_name = ?
            """, (phone_name,))
            
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                phone_data = CrowdsourcedPhoneData(
                    phone_name=phone_name,
                    aggregated_rating=data.get('aggregated_rating'),
                    user_reviews=data.get('user_reviews', []),
                    specifications=data.get('specifications', {}),
                    price_data=data.get('price_data', {}),
                    pros_cons=data.get('pros_cons', {'pros': [], 'cons': []}),
                    usage_experiences=data.get('usage_experiences', []),
                    common_issues=data.get('common_issues', []),
                    recommended_alternatives=data.get('recommended_alternatives', []),
                    community_confidence=data.get('community_confidence', 0.0),
                    last_updated=row[1],
                    contributor_count=row[2]
                )
                
                self.phone_data_cache[phone_name] = phone_data
                return phone_data
        
        return None
    
    def get_user_contributions(self, user_id: str, limit: int = 50) -> List[UserContribution]:
        """Get contributions by a specific user"""
        
        contributions = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM contributions 
                WHERE user_id = ? 
                ORDER BY submitted_at DESC 
                LIMIT ?
            """, (user_id, limit))
            
            for row in cursor.fetchall():
                contribution = self._row_to_contribution(row)
                contributions.append(contribution)
        
        return contributions
    
    def get_pending_moderation_tasks(self, moderator_id: str = None) -> List[ModerationTask]:
        """Get pending moderation tasks"""
        
        tasks = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if moderator_id:
                cursor.execute("""
                    SELECT * FROM moderation_tasks 
                    WHERE completed = 0 AND (assigned_moderator = ? OR assigned_moderator IS NULL)
                    ORDER BY priority DESC, created_at ASC
                """, (moderator_id,))
            else:
                cursor.execute("""
                    SELECT * FROM moderation_tasks 
                    WHERE completed = 0 
                    ORDER BY priority DESC, created_at ASC
                """)
            
            for row in cursor.fetchall():
                task = ModerationTask(
                    task_id=row[0],
                    contribution_id=row[1],
                    priority=row[2],
                    assigned_moderator=row[3],
                    created_at=row[4],
                    due_date=row[5],
                    notes=row[6],
                    completed=bool(row[7])
                )
                tasks.append(task)
        
        return tasks
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        
        if user_id in self.user_cache:
            return self.user_cache[user_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                user = UserProfile(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    role=UserRole(row[3]),
                    reputation_score=row[4],
                    total_contributions=row[5],
                    approved_contributions=row[6],
                    expertise_areas=json.loads(row[7]) if row[7] else [],
                    joined_at=row[8],
                    last_active=row[9],
                    preferences=json.loads(row[10]) if row[10] else {},
                    achievements=json.loads(row[11]) if row[11] else []
                )
                
                self.user_cache[user_id] = user
                return user
        
        return None
    
    def get_contribution(self, contribution_id: str) -> Optional[UserContribution]:
        """Get a specific contribution"""
        
        if contribution_id in self.contribution_cache:
            return self.contribution_cache[contribution_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM contributions WHERE contribution_id = ?", (contribution_id,))
            row = cursor.fetchone()
            
            if row:
                contribution = self._row_to_contribution(row)
                self.contribution_cache[contribution_id] = contribution
                return contribution
        
        return None
    
    def search_contributions(self, phone_name: str = None, user_id: str = None, 
                           contribution_type: ContributionType = None, 
                           status: ContributionStatus = None, limit: int = 100) -> List[UserContribution]:
        """Search contributions with filters"""
        
        contributions = []
        conditions = []
        params = []
        
        if phone_name:
            conditions.append("phone_name LIKE ?")
            params.append(f"%{phone_name}%")
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if contribution_type:
            conditions.append("contribution_type = ?")
            params.append(contribution_type.value)
        
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT * FROM contributions 
                {where_clause}
                ORDER BY submitted_at DESC 
                LIMIT ?
            """, params)
            
            for row in cursor.fetchall():
                contribution = self._row_to_contribution(row)
                contributions.append(contribution)
        
        return contributions
    
    def get_community_statistics(self) -> Dict[str, Any]:
        """Get community statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # User statistics
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE last_active > ?", 
                          ((datetime.now() - timedelta(days=30)).isoformat(),))
            active_users = cursor.fetchone()[0]
            
            # Contribution statistics
            cursor.execute("SELECT COUNT(*) FROM contributions")
            total_contributions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM contributions WHERE status = ?", 
                          (ContributionStatus.APPROVED.value,))
            approved_contributions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM contributions WHERE status = ?", 
                          (ContributionStatus.PENDING.value,))
            pending_contributions = cursor.fetchone()[0]
            
            # Phone coverage
            cursor.execute("SELECT COUNT(DISTINCT phone_name) FROM contributions")
            phones_covered = cursor.fetchone()[0]
            
            # Top contributors
            cursor.execute("""
                SELECT username, reputation_score, approved_contributions 
                FROM users 
                ORDER BY reputation_score DESC 
                LIMIT 10
            """)
            top_contributors = [
                {'username': row[0], 'reputation': row[1], 'contributions': row[2]}
                for row in cursor.fetchall()
            ]
            
            return {
                'total_users': total_users,
                'active_users_30d': active_users,
                'total_contributions': total_contributions,
                'approved_contributions': approved_contributions,
                'pending_contributions': pending_contributions,
                'phones_covered': phones_covered,
                'approval_rate': approved_contributions / total_contributions if total_contributions > 0 else 0,
                'top_contributors': top_contributors
            }
    
    def _calculate_contribution_confidence(self, user: UserProfile, data: Dict[str, Any], 
                                         contribution_type: ContributionType) -> float:
        """Calculate confidence score for a contribution"""
        
        score = 0.5  # Base score
        
        # User reputation factor
        reputation_factor = min(user.reputation_score / 100, 0.3)
        score += reputation_factor
        
        # Data completeness
        data_completeness = min(len(data) / 10, 0.2)
        score += data_completeness
        
        # Expertise area match
        if contribution_type == ContributionType.SPECIFICATION and 'technical' in user.expertise_areas:
            score += 0.1
        elif contribution_type == ContributionType.REVIEW and 'reviews' in user.expertise_areas:
            score += 0.1
        
        # Data quality indicators
        if contribution_type == ContributionType.REVIEW:
            review_text = data.get('review_text', '')
            if len(review_text) > 200:
                score += 0.1
            if data.get('pros') and data.get('cons'):
                score += 0.1
        
        return min(score, 1.0)
    
    def _update_user_reputation(self, user_id: str, action: str):
        """Update user reputation based on action"""
        
        change = self.config['reputation_weights'].get(action, 0)
        
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE users SET reputation_score = reputation_score + ?
                    WHERE user_id = ?
                """, (change, user_id))
                
                conn.commit()
        
        # Update cache
        if user_id in self.user_cache:
            self.user_cache[user_id].reputation_score += change
        
        # Check for role promotions
        self._check_role_promotions(user_id)
    
    def _check_role_promotions(self, user_id: str):
        """Check if user should be promoted to higher role"""
        
        user = self.get_user_profile(user_id)
        if not user:
            return
        
        # Promote to Expert if conditions met
        if (user.role == UserRole.CONTRIBUTOR 
            and user.reputation_score >= 200 
            and user.approved_contributions >= 50):
            
            self._promote_user(user_id, UserRole.EXPERT)
            self._add_achievement(user_id, "Expert Reviewer")
        
        # Promote to Moderator (manual process, but we can flag candidates)
        elif (user.role == UserRole.EXPERT 
              and user.reputation_score >= 500 
              and user.approved_contributions >= 100):
            
            logger.info(f"User {user.username} is eligible for moderator role")
    
    def _promote_user(self, user_id: str, new_role: UserRole):
        """Promote user to new role"""
        
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE users SET role = ? WHERE user_id = ?
                """, (new_role.value, user_id))
                
                conn.commit()
        
        if user_id in self.user_cache:
            self.user_cache[user_id].role = new_role
        
        logger.info(f"Promoted user {user_id} to {new_role.value}")
    
    def _add_achievement(self, user_id: str, achievement: str):
        """Add achievement to user"""
        
        user = self.get_user_profile(user_id)
        if user and achievement not in user.achievements:
            user.achievements.append(achievement)
            
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        UPDATE users SET achievements = ? WHERE user_id = ?
                    """, (json.dumps(user.achievements), user_id))
                    
                    conn.commit()
    
    def _create_moderation_task(self, contribution_id: str):
        """Create a moderation task for a contribution"""
        
        task_id = str(uuid.uuid4())
        priority = 3  # Default priority
        due_date = (datetime.now() + timedelta(days=7)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO moderation_tasks 
                (task_id, contribution_id, priority, created_at, due_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (task_id, contribution_id, priority, datetime.now().isoformat(), due_date, ""))
            
            conn.commit()
    
    def _check_vote_based_approval(self, contribution_id: str):
        """Check if contribution should be approved based on votes"""
        
        contribution = self.get_contribution(contribution_id)
        if not contribution or contribution.status != ContributionStatus.PENDING:
            return
        
        if contribution.votes_up >= self.config['votes_required_for_approval']:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        UPDATE contributions 
                        SET status = ?, reviewed_at = ?, reviewer_id = ?
                        WHERE contribution_id = ?
                    """, (ContributionStatus.APPROVED.value, datetime.now().isoformat(), 
                          "community_approval", contribution_id))
                    
                    cursor.execute("""
                        UPDATE users SET approved_contributions = approved_contributions + 1
                        WHERE user_id = ?
                    """, (contribution.user_id,))
                    
                    conn.commit()
            
            self._update_user_reputation(contribution.user_id, 'approved_contribution')
            self._update_phone_data_aggregation(contribution.phone_name)
    
    def _count_pending_contributions(self, user_id: str) -> int:
        """Count pending contributions for a user"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM contributions 
                WHERE user_id = ? AND status = ?
            """, (user_id, ContributionStatus.PENDING.value))
            
            return cursor.fetchone()[0]
    
    def _update_phone_data_aggregation(self, phone_name: str):
        """Update aggregated phone data"""
        
        # Get all approved contributions for this phone
        contributions = self.search_contributions(
            phone_name=phone_name,
            status=ContributionStatus.APPROVED
        )
        
        # Aggregate data
        aggregated_data = self._aggregate_phone_contributions(contributions)
        
        # Save to database
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO phone_data_aggregations 
                    (phone_name, aggregated_data, last_updated, contributor_count)
                    VALUES (?, ?, ?, ?)
                """, (
                    phone_name, 
                    json.dumps(aggregated_data),
                    datetime.now().isoformat(),
                    len(set(c.user_id for c in contributions))
                ))
                
                conn.commit()
        
        # Update cache
        phone_data = CrowdsourcedPhoneData(
            phone_name=phone_name,
            **aggregated_data,
            last_updated=datetime.now().isoformat(),
            contributor_count=len(set(c.user_id for c in contributions))
        )
        self.phone_data_cache[phone_name] = phone_data
    
    def _aggregate_phone_contributions(self, contributions: List[UserContribution]) -> Dict[str, Any]:
        """Aggregate contributions into phone data"""
        
        aggregated = {
            'aggregated_rating': None,
            'user_reviews': [],
            'specifications': {},
            'price_data': {},
            'pros_cons': {'pros': [], 'cons': []},
            'usage_experiences': [],
            'common_issues': [],
            'recommended_alternatives': [],
            'community_confidence': 0.0
        }
        
        ratings = []
        all_pros = []
        all_cons = []
        
        for contrib in contributions:
            data = contrib.data
            
            # Collect ratings
            if contrib.contribution_type == ContributionType.RATING and 'rating' in data:
                ratings.append(data['rating'])
            
            # Collect reviews
            if contrib.contribution_type == ContributionType.REVIEW:
                review = {
                    'author': contrib.user_id,
                    'rating': data.get('rating'),
                    'title': data.get('title', ''),
                    'content': data.get('review_text', ''),
                    'pros': data.get('pros', []),
                    'cons': data.get('cons', []),
                    'submitted_at': contrib.submitted_at
                }
                aggregated['user_reviews'].append(review)
                
                if data.get('pros'):
                    all_pros.extend(data['pros'])
                if data.get('cons'):
                    all_cons.extend(data['cons'])
            
            # Collect specifications
            if contrib.contribution_type == ContributionType.SPECIFICATION:
                for key, value in data.items():
                    if key not in aggregated['specifications']:
                        aggregated['specifications'][key] = []
                    aggregated['specifications'][key].append(value)
            
            # Collect price data
            if contrib.contribution_type == ContributionType.PRICE_UPDATE:
                price_key = f"{data.get('retailer', 'unknown')}_{data.get('variant', 'default')}"
                aggregated['price_data'][price_key] = {
                    'price': data.get('price'),
                    'currency': data.get('currency', 'USD'),
                    'last_updated': contrib.submitted_at,
                    'retailer': data.get('retailer', 'Unknown')
                }
        
        # Calculate aggregated rating
        if ratings:
            aggregated['aggregated_rating'] = statistics.mean(ratings)
        
        # Aggregate pros and cons
        if all_pros:
            pro_counts = defaultdict(int)
            for pro in all_pros:
                pro_counts[pro] += 1
            aggregated['pros_cons']['pros'] = [
                pro for pro, count in sorted(pro_counts.items(), key=lambda x: x[1], reverse=True)
            ][:10]  # Top 10
        
        if all_cons:
            con_counts = defaultdict(int)
            for con in all_cons:
                con_counts[con] += 1
            aggregated['pros_cons']['cons'] = [
                con for con, count in sorted(con_counts.items(), key=lambda x: x[1], reverse=True)
            ][:10]  # Top 10
        
        # Calculate community confidence
        if contributions:
            avg_confidence = statistics.mean([c.confidence_score for c in contributions])
            contributor_diversity = len(set(c.user_id for c in contributions))
            aggregated['community_confidence'] = min(avg_confidence * (contributor_diversity / 10), 1.0)
        
        return aggregated
    
    def _row_to_contribution(self, row) -> UserContribution:
        """Convert database row to UserContribution object"""
        
        return UserContribution(
            contribution_id=row[0],
            user_id=row[1],
            contribution_type=ContributionType(row[2]),
            phone_name=row[3],
            data=json.loads(row[4]),
            status=ContributionStatus(row[5]),
            submitted_at=row[6],
            reviewed_at=row[7],
            reviewer_id=row[8],
            confidence_score=row[9],
            votes_up=row[10],
            votes_down=row[11],
            comments=json.loads(row[12]) if row[12] else [],
            metadata=json.loads(row[13]) if row[13] else {}
        )
    
    def _load_caches(self):
        """Load data into memory caches"""
        
        try:
            # Load recent users
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM users 
                    WHERE last_active > ? 
                    ORDER BY last_active DESC 
                    LIMIT 100
                """, ((datetime.now() - timedelta(days=7)).isoformat(),))
                
                for row in cursor.fetchall():
                    user = UserProfile(
                        user_id=row[0],
                        username=row[1],
                        email=row[2],
                        role=UserRole(row[3]),
                        reputation_score=row[4],
                        total_contributions=row[5],
                        approved_contributions=row[6],
                        expertise_areas=json.loads(row[7]) if row[7] else [],
                        joined_at=row[8],
                        last_active=row[9],
                        preferences=json.loads(row[10]) if row[10] else {},
                        achievements=json.loads(row[11]) if row[11] else []
                    )
                    self.user_cache[user.user_id] = user
            
            logger.info(f"Loaded {len(self.user_cache)} users into cache")
            
        except Exception as e:
            logger.error(f"Failed to load caches: {e}")

# Factory function
def create_user_feedback_system(config=None):
    """Create configured user feedback system"""
    return UserFeedbackSystem(config=config)