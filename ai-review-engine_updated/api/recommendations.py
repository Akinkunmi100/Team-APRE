"""
Recommendation API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from database.models import db_manager, Product, Review, User
from models.recommendation_engine import recommendation_engine
from api.main import get_current_user, rate_limit

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


def get_current_user_optional(credentials = None):
    """Get current user optionally (for public endpoints)"""
    if not credentials:
        return None
    try:
        from api.main import get_current_user
        return get_current_user(credentials)
    except:
        return None


@router.get("/for-product/{product_id}")
async def get_product_recommendations(
    product_id: int,
    limit: int = Query(10, ge=1, le=50),
    strategy: str = Query("hybrid", regex="^(content|collaborative|popular|hybrid)$"),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get recommendations based on a specific product"""
    
    # Check if product exists
    session = db_manager.get_session()
    product = session.query(Product).filter_by(id=product_id).first()
    session.close()
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    user_id = current_user.id if current_user else None
    
    recommendations = recommendation_engine.get_recommendations(
        product_id=product_id,
        user_id=user_id,
        n_recommendations=limit,
        strategy=strategy
    )
    
    return {
        "base_product": {
            "id": product.id,
            "name": product.name,
            "brand": product.brand
        },
        "recommendations": recommendations,
        "strategy_used": strategy,
        "count": len(recommendations)
    }


@router.get("/for-user")
@rate_limit(max_calls=100)
async def get_user_recommendations(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """Get personalized recommendations for the current user"""
    
    recommendations = recommendation_engine.get_personalized_feed(
        user_id=current_user.id,
        limit=limit
    )
    
    return {
        "user_id": current_user.id,
        "recommendations": recommendations,
        "count": len(recommendations),
        "generated_at": datetime.now().isoformat()
    }


@router.get("/similar-products")
async def get_similar_products(
    product_id: int,
    limit: int = Query(10, ge=1, le=30)
):
    """Get products similar to the specified product"""
    
    recommendations = recommendation_engine.get_recommendations(
        product_id=product_id,
        n_recommendations=limit,
        strategy='content'
    )
    
    return {
        "similar_products": recommendations,
        "count": len(recommendations)
    }


@router.get("/trending")
async def get_trending_products(
    limit: int = Query(20, ge=1, le=50),
    days: int = Query(7, ge=1, le=30)
):
    """Get trending products based on recent activity"""
    
    recommendations = recommendation_engine._get_trending_recommendations(limit)
    
    return {
        "trending_products": recommendations,
        "period_days": days,
        "count": len(recommendations)
    }


@router.get("/popular")
async def get_popular_products(
    limit: int = Query(20, ge=1, le=50),
    category: Optional[str] = None
):
    """Get popular products overall or by category"""
    
    if category:
        recommendations = recommendation_engine._get_category_recommendations(
            category=category,
            n=limit
        )
    else:
        recommendations = recommendation_engine._get_popular_recommendations(limit)
    
    return {
        "popular_products": recommendations,
        "category": category,
        "count": len(recommendations)
    }


@router.post("/search-based")
async def get_search_based_recommendations(
    query: str,
    limit: int = Query(10, ge=1, le=50),
    include_product_id: Optional[int] = None,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get recommendations based on search query"""
    
    user_id = current_user.id if current_user else None
    
    recommendations = recommendation_engine.get_recommendations(
        product_id=include_product_id,
        user_id=user_id,
        search_query=query,
        n_recommendations=limit,
        strategy='hybrid'
    )
    
    return {
        "query": query,
        "recommendations": recommendations,
        "count": len(recommendations)
    }


@router.post("/update-models")
@rate_limit(max_calls=1)
async def update_recommendation_models(
    current_user: User = Depends(get_current_user)
):
    """Update recommendation models with latest data (admin only)"""
    
    # Check if user is admin (you can add an is_admin field to User model)
    # For now, we'll just require authentication
    
    try:
        recommendation_engine.update_models()
        return {
            "status": "success",
            "message": "Recommendation models updated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to update recommendation models: {e}")
        raise HTTPException(status_code=500, detail="Failed to update models")


@router.get("/categories")
async def get_recommendation_categories():
    """Get available product categories for recommendations"""
    
    session = db_manager.get_session()
    
    # Get distinct categories
    categories = session.query(Product.category).distinct().all()
    
    session.close()
    
    return {
        "categories": [c[0] for c in categories if c[0]],
        "count": len(categories)
    }


@router.get("/bundle")
async def get_product_bundle_recommendations(
    product_ids: str = Query(..., description="Comma-separated product IDs"),
    limit: int = Query(5, ge=1, le=20)
):
    """Get recommendations for products that go well together (bundle recommendations)"""
    
    # Parse product IDs
    try:
        ids = [int(id.strip()) for id in product_ids.split(',')]
    except:
        raise HTTPException(status_code=400, detail="Invalid product IDs format")
    
    if len(ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 products allowed")
    
    # Get recommendations for each product and find common ones
    all_recommendations = []
    
    for pid in ids:
        recs = recommendation_engine.get_recommendations(
            product_id=pid,
            n_recommendations=limit * 2,
            strategy='content'
        )
        all_recommendations.extend(recs)
    
    # Aggregate and rank
    product_scores = {}
    for rec in all_recommendations:
        pid = rec['product_id']
        if pid not in ids:  # Don't recommend products already in bundle:
            if pid not in product_scores:
                product_scores[pid] = {
                    'data': rec,
                    'score': 0,
                    'count': 0
                }
            product_scores[pid]['score'] += rec['score']
            product_scores[pid]['count'] += 1
    
    # Sort by combined score
    bundle_recommendations = []
    for pid, info in product_scores.items():
        rec = info['data'].copy()
        rec['bundle_score'] = info['score'] / info['count'] * (1 + 0.2 * info['count'])
        rec['reason'] = f"Complements your selection ({info['count']} matches)"
        bundle_recommendations.append(rec)
    
    bundle_recommendations.sort(key=lambda x: x['bundle_score'], reverse=True)
    
    return {
        "bundle_products": ids,
        "recommendations": bundle_recommendations[:limit],
        "count": min(len(bundle_recommendations), limit)
    }


# Export router to be included in main app
def include_recommendation_routes(app):
    """Include recommendation routes in the main app"""
    app.include_router(router)
