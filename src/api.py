"""
FastAPI application for the University Matching Algorithm.
Provides RESTful API endpoints for university recommendations.
"""
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
from fastapi.responses import JSONResponse

from .models import (
    StudentProfile, University, RecommendationRequest, RecommendationResponse,
    FilterCriteria, UniversityType
)
from .recommendation_engine import UniversityRecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="University Matching Algorithm API",
    description="An intelligent recommendation system for matching students with universities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommendation engine instance
recommendation_engine: Optional[UniversityRecommendationEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine on startup."""
    global recommendation_engine
    try:
        logger.info("Initializing University Recommendation Engine...")
        recommendation_engine = UniversityRecommendationEngine()
        if recommendation_engine.is_initialized:
            logger.info("Recommendation engine initialized successfully")
        else:
            logger.error("Failed to initialize recommendation engine")
    except Exception as e:
        logger.error(f"Error during startup: {e}")


def get_recommendation_engine() -> UniversityRecommendationEngine:
    """Dependency to get the recommendation engine instance."""
    if recommendation_engine is None or not recommendation_engine.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation engine not available. Please try again later."
        )
    return recommendation_engine


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "University Matching Algorithm API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "recommendations": "/recommendations",
            "universities": "/universities",
            "university_by_id": "/universities/{id}",
            "filter_universities": "/universities/filter",
            "statistics": "/statistics",
            "documentation": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check(engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)):
    """Health check endpoint."""
    stats = engine.get_dataset_statistics()
    return {
        "status": "healthy",
        "engine_initialized": engine.is_initialized,
        "total_universities": stats.get("total_universities", 0),
        "available_countries": len(stats.get("countries", [])),
        "ml_models_trained": engine.ml_recommender.is_trained
    }


@app.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get university recommendations for a student profile.
    
    This is the main endpoint that analyzes a student's profile and returns
    ranked university recommendations with detailed explanations.
    """
    try:
        # Validate student profile
        warnings = engine.validate_student_profile(request.student_profile)
        if warnings:
            logger.warning(f"Student profile validation warnings: {warnings}")
        
        # Generate recommendations
        response = engine.get_recommendations(
            student_profile=request.student_profile,
            max_recommendations=request.max_recommendations,
            include_explanations=request.include_explanations
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.post("/recommendations/validate", tags=["Recommendations"])
async def validate_student_profile(
    student_profile: StudentProfile,
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Validate a student profile and return warnings or suggestions.
    """
    try:
        warnings = engine.validate_student_profile(student_profile)
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "profile_summary": {
                "gpa": student_profile.gpa,
                "budget": student_profile.budget_max,
                "field_of_study": student_profile.field_of_study,
                "preferred_countries": student_profile.preferred_countries,
                "test_scores": student_profile.test_scores
            }
        }
        
    except Exception as e:
        logger.error(f"Error validating student profile: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating profile: {str(e)}")


@app.get("/universities", response_model=List[University], tags=["Universities"])
async def get_universities(
    limit: Optional[int] = Query(50, ge=1, le=500, description="Maximum number of universities to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of universities to skip"),
    country: Optional[str] = Query(None, description="Filter by country"),
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get a list of all universities in the dataset with optional filtering.
    """
    try:
        all_universities = engine.get_all_universities()
        
        # Apply country filter if specified
        if country:
            all_universities = [u for u in all_universities if u.country.lower() == country.lower()]
        
        # Apply pagination
        total = len(all_universities)
        universities = all_universities[offset or 0:(offset or 0) + (limit or 50)]
        
        # Add pagination info to response headers
        return universities
        
    except Exception as e:
        logger.error(f"Error retrieving universities: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving universities: {str(e)}")


@app.get("/universities/{university_id}", response_model=University, tags=["Universities"])
async def get_university_by_id(
    university_id: str,
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get detailed information about a specific university.
    """
    try:
        university = engine.get_university_by_id(university_id)
        
        if university is None:
            raise HTTPException(status_code=404, detail=f"University with ID '{university_id}' not found")
        
        return university
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving university {university_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving university: {str(e)}")


@app.post("/universities/filter", response_model=List[University], tags=["Universities"])
async def filter_universities(
    filter_criteria: FilterCriteria,
    limit: Optional[int] = Query(100, ge=1, le=500, description="Maximum number of results"),
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Filter universities based on specific criteria.
    """
    try:
        filtered_universities = engine.get_filtered_universities(filter_criteria)
        
        # Apply limit
        universities = filtered_universities[:limit]
        
        return universities
        
    except Exception as e:
        logger.error(f"Error filtering universities: {e}")
        raise HTTPException(status_code=500, detail=f"Error filtering universities: {str(e)}")


@app.get("/statistics", tags=["Analytics"])
async def get_dataset_statistics(
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get statistics about the university dataset.
    """
    try:
        stats = engine.get_dataset_statistics()
        
        # Add some additional computed statistics
        enhanced_stats = {
            **stats,
            "api_version": "1.0.0",
            "engine_status": "active" if engine.is_initialized else "inactive",
            "ml_models_available": engine.ml_recommender.is_trained
        }
        
        return enhanced_stats
        
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")


@app.get("/countries", tags=["Analytics"])
async def get_available_countries(
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get a list of all countries available in the dataset.
    """
    try:
        stats = engine.get_dataset_statistics()
        countries = stats.get("countries", [])
        
        # Sort countries alphabetically
        countries.sort()
        
        return {
            "countries": countries,
            "total_count": len(countries)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving countries: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving countries: {str(e)}")


@app.get("/programs", tags=["Analytics"])
async def get_available_programs(
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get a list of all academic programs available in the dataset.
    """
    try:
        stats = engine.get_dataset_statistics()
        programs = stats.get("programs_available", [])
        
        # Sort programs alphabetically
        programs.sort()
        
        return {
            "programs": programs,
            "total_count": len(programs)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving programs: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving programs: {str(e)}")


@app.post("/admin/refresh", tags=["Admin"])
async def refresh_data(
    engine: UniversityRecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Refresh the university dataset and retrain ML models.
    This endpoint is for administrative use to update the system with new data.
    """
    try:
        success = engine.refresh_data()
        
        if success:
            stats = engine.get_dataset_statistics()
            return {
                "status": "success",
                "message": "Data refresh completed successfully",
                "total_universities": stats.get("total_universities", 0),
                "ml_models_retrained": engine.ml_recommender.is_trained
            }
        else:
            raise HTTPException(status_code=500, detail="Data refresh failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": f"The requested resource was not found: {request.url.path}"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 