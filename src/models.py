"""
Data models and schemas for the University Matching Algorithm.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class UniversityType(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    TECHNICAL = "technical"


class CampusCulture(str, Enum):
    RESEARCH_FOCUSED = "research_focused"
    LIBERAL_ARTS = "liberal_arts"
    TECHNICAL = "technical"
    DIVERSE = "diverse"
    TRADITIONAL = "traditional"
    MODERN = "modern"


class StudentProfile(BaseModel):
    """Student profile data model."""
    
    # Academic Information
    gpa: float = Field(..., ge=0.0, le=4.0, description="GPA on 4.0 scale")
    test_scores: Dict[str, int] = Field(default_factory=dict, description="Test scores (SAT, TOEFL, etc.)")
    qualifications: List[str] = Field(default_factory=list, description="Academic qualifications and certifications")
    
    # Preferences
    preferred_countries: List[str] = Field(default_factory=list, description="Preferred countries")
    preferred_cities: List[str] = Field(default_factory=list, description="Preferred cities")
    budget_max: float = Field(..., gt=0, description="Maximum budget in USD")
    field_of_study: str = Field(..., description="Intended field of study")
    
    # Constraints
    language_requirements: Dict[str, str] = Field(default_factory=dict, description="Language proficiency levels")
    timeline: Optional[str] = Field(None, description="Intended start date")
    
    # Priorities (weights for scoring)
    priorities: Dict[str, float] = Field(
        default_factory=lambda: {
            "academic_reputation": 0.3,
            "cost": 0.25,
            "location": 0.2,
            "culture": 0.15,
            "admission_probability": 0.1
        },
        description="Priority weights for different factors"
    )
    
    # Personal Preferences
    university_size_preference: Optional[str] = Field(None, description="Small, Medium, Large")
    university_type_preference: Optional[UniversityType] = None
    campus_culture_preferences: List[CampusCulture] = Field(default_factory=list)


class University(BaseModel):
    """University data model."""
    
    # Basic Information
    university_id: str = Field(..., description="Unique university identifier")
    name: str = Field(..., description="University name")
    country: str = Field(..., description="Country")
    city: str = Field(..., description="City")
    
    # Rankings and Reputation
    ranking_global: Optional[int] = Field(None, description="Global ranking")
    ranking_national: Optional[int] = Field(None, description="National ranking")
    reputation_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Reputation score")
    
    # Financial Information
    tuition_fees_usd: float = Field(..., ge=0, description="Annual tuition fees in USD")
    living_cost_usd: float = Field(..., ge=0, description="Annual living costs in USD")
    
    # Admission Requirements
    min_gpa: float = Field(..., ge=0.0, le=4.0, description="Minimum GPA requirement")
    min_test_scores: Dict[str, int] = Field(default_factory=dict, description="Minimum test score requirements")
    acceptance_rate: float = Field(..., ge=0.0, le=1.0, description="Acceptance rate as decimal")
    
    # Demographics and Culture
    international_student_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="International student ratio")
    student_population: int = Field(default=0, ge=0, description="Total student population")
    
    # Academic Information
    programs_offered: List[str] = Field(default_factory=list, description="Available programs")
    languages_taught: List[str] = Field(default_factory=list, description="Languages of instruction")
    
    # University Characteristics
    university_type: UniversityType = Field(default=UniversityType.PUBLIC)
    campus_culture_tags: List[CampusCulture] = Field(default_factory=list)
    
    # Additional Features
    has_scholarship: bool = Field(default=False, description="Offers scholarships")
    research_opportunities: bool = Field(default=False, description="Research opportunities available")


class MatchScore(BaseModel):
    """Individual scoring component for a university match."""
    
    academic_fit: float = Field(..., ge=0.0, le=100.0, description="Academic compatibility score")
    financial_feasibility: float = Field(..., ge=0.0, le=100.0, description="Financial feasibility score")
    preference_alignment: float = Field(..., ge=0.0, le=100.0, description="Preference alignment score")
    admission_probability: float = Field(..., ge=0.0, le=100.0, description="Admission probability score")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Weighted overall score")


class Recommendation(BaseModel):
    """University recommendation with detailed scoring and explanation."""
    
    university: University = Field(..., description="University details")
    match_score: MatchScore = Field(..., description="Detailed scoring breakdown")
    explanation: str = Field(..., description="Why this university was recommended")
    concerns: List[str] = Field(default_factory=list, description="Potential concerns or challenges")
    total_annual_cost: float = Field(..., description="Total annual cost (tuition + living)")
    rank_position: int = Field(..., description="Position in recommendation ranking")


class RecommendationRequest(BaseModel):
    """Request model for getting university recommendations."""
    
    student_profile: StudentProfile
    max_recommendations: int = Field(default=20, ge=1, le=50, description="Maximum number of recommendations")
    include_explanations: bool = Field(default=True, description="Include detailed explanations")


class RecommendationResponse(BaseModel):
    """Response model for university recommendations."""
    
    recommendations: List[Recommendation] = Field(..., description="List of university recommendations")
    total_universities_considered: int = Field(..., description="Total universities in dataset")
    universities_after_filtering: int = Field(..., description="Universities remaining after filtering")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    student_summary: Dict[str, Any] = Field(..., description="Summary of student profile used")


class FilterCriteria(BaseModel):
    """Criteria used for filtering universities."""
    
    max_tuition: Optional[float] = None
    max_total_cost: Optional[float] = None
    min_gpa_requirement: Optional[float] = None
    countries: Optional[List[str]] = None
    cities: Optional[List[str]] = None
    programs: Optional[List[str]] = None
    university_types: Optional[List[UniversityType]] = None
    min_acceptance_rate: Optional[float] = None
    max_acceptance_rate: Optional[float] = None 