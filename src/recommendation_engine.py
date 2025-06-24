"""
Main recommendation engine that orchestrates all components.
Generates university recommendations with detailed explanations.
"""
import time
import logging
from typing import List, Dict, Any, Optional

from .models import (
    StudentProfile, University, MatchScore, Recommendation, 
    RecommendationResponse, FilterCriteria
)
from .data_processor import UniversityDataProcessor
from .matching_algorithm import UniversityMatchingAlgorithm
from .ml_models import HybridRecommender

logger = logging.getLogger(__name__)


class UniversityRecommendationEngine:
    """Main recommendation engine for university matching."""
    
    def __init__(self, data_file_path: str = "data/universities.csv"):
        """Initialize the recommendation engine.
        
        Args:
            data_file_path: Path to university dataset
        """
        self.data_processor = UniversityDataProcessor(data_file_path)
        self.matching_algorithm = UniversityMatchingAlgorithm()
        self.ml_recommender = HybridRecommender()
        self.is_initialized = False
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the recommendation engine components."""
        try:
            universities = self.data_processor.get_universities()
            
            if not universities:
                logger.error("No universities loaded")
                return
            
            # Train ML models
            logger.info("Training machine learning models...")
            self.ml_recommender.train(universities)
            
            self.is_initialized = True
            logger.info(f"Recommendation engine initialized with {len(universities)} universities")
            
        except Exception as e:
            logger.error(f"Error initializing recommendation engine: {e}")
            self.is_initialized = False
    
    def get_recommendations(self, 
                          student_profile: StudentProfile,
                          max_recommendations: int = 20,
                          include_explanations: bool = True) -> RecommendationResponse:
        """Generate university recommendations for a student.
        
        Args:
            student_profile: Student profile with preferences and constraints
            max_recommendations: Maximum number of recommendations to return
            include_explanations: Whether to include detailed explanations
            
        Returns:
            RecommendationResponse with ranked university recommendations
        """
        start_time = time.time()
        
        try:
            # Get all universities
            all_universities = self.data_processor.get_universities()
            total_universities = len(all_universities)
            
            if not all_universities:
                return RecommendationResponse(
                    recommendations=[],
                    total_universities_considered=0,
                    universities_after_filtering=0,
                    processing_time_ms=0,
                    student_summary=self._create_student_summary(student_profile)
                )
            
            # Step 1: Apply hard constraint filters
            logger.info("Applying hard constraint filters...")
            filtered_universities = self.matching_algorithm.apply_hard_filters(
                all_universities, student_profile
            )
            universities_after_filtering = len(filtered_universities)
            
            if not filtered_universities:
                logger.warning("No universities match the hard constraints")
                return RecommendationResponse(
                    recommendations=[],
                    total_universities_considered=total_universities,
                    universities_after_filtering=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    student_summary=self._create_student_summary(student_profile)
                )
            
            # Step 2: Calculate traditional matching scores
            logger.info("Calculating traditional matching scores...")
            scored_matches = self.matching_algorithm.calculate_match_scores(
                filtered_universities, student_profile
            )
            
            # Step 3: Apply ML recommendations if available
            if self.ml_recommender.is_trained and len(filtered_universities) > 5:  # Only use ML if enough data
                logger.info("Applying machine learning recommendations...")
                try:
                    ml_recommendations = self.ml_recommender.recommend(
                        student_profile, filtered_universities, top_k=len(filtered_universities)
                    )
                    
                    # Combine traditional and ML scores
                    combined_scores = self._combine_scores(scored_matches, ml_recommendations)
                except Exception as e:
                    logger.warning(f"ML recommendation failed, using traditional scoring: {e}")
                    combined_scores = scored_matches
            else:
                logger.info("Using traditional scoring only")
                combined_scores = scored_matches
            
            # Step 4: Generate final recommendations
            recommendations = []
            for rank, (university, match_score) in enumerate(combined_scores[:max_recommendations], 1):
                # Generate explanation if requested
                explanation = ""
                concerns = []
                
                if include_explanations:
                    explanation = self.matching_algorithm.generate_explanation(
                        university, match_score, student_profile
                    )
                    concerns = self.matching_algorithm.identify_concerns(
                        university, student_profile
                    )
                
                # Calculate total annual cost
                total_cost = university.tuition_fees_usd + university.living_cost_usd
                
                recommendation = Recommendation(
                    university=university,
                    match_score=match_score,
                    explanation=explanation,
                    concerns=concerns,
                    total_annual_cost=total_cost,
                    rank_position=rank
                )
                
                recommendations.append(recommendation)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Generated {len(recommendations)} recommendations in {processing_time_ms:.2f}ms")
            
            return RecommendationResponse(
                recommendations=recommendations,
                total_universities_considered=total_universities,
                universities_after_filtering=universities_after_filtering,
                processing_time_ms=processing_time_ms,
                student_summary=self._create_student_summary(student_profile)
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            
            return RecommendationResponse(
                recommendations=[],
                total_universities_considered=0,
                universities_after_filtering=0,
                processing_time_ms=processing_time_ms,
                student_summary=self._create_student_summary(student_profile)
            )
    
    def _combine_scores(self, 
                       traditional_scores: List[tuple], 
                       ml_scores: List[tuple],
                       traditional_weight: float = 0.7,
                       ml_weight: float = 0.3) -> List[tuple]:
        """Combine traditional matching scores with ML scores.
        
        Args:
            traditional_scores: List of (University, MatchScore) from traditional algorithm
            ml_scores: List of (University, float) from ML models
            traditional_weight: Weight for traditional scores
            ml_weight: Weight for ML scores
            
        Returns:
            Combined scored matches sorted by final score
        """
        # Create lookup dictionary for ML scores
        ml_score_dict = {univ.university_id: score for univ, score in ml_scores}
        
        combined_matches = []
        
        for university, match_score in traditional_scores:
            # Get ML score (normalized to 0-100 scale)
            ml_score = ml_score_dict.get(university.university_id, 0.0) * 100
            
            # Combine traditional overall score with ML score
            combined_overall_score = (
                traditional_weight * match_score.overall_score +
                ml_weight * ml_score
            )
            
            # Update the match score with combined overall score
            updated_match_score = MatchScore(
                academic_fit=match_score.academic_fit,
                financial_feasibility=match_score.financial_feasibility,
                preference_alignment=match_score.preference_alignment,
                admission_probability=match_score.admission_probability,
                overall_score=combined_overall_score
            )
            
            combined_matches.append((university, updated_match_score))
        
        # Sort by combined overall score
        combined_matches.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return combined_matches
    
    def _create_student_summary(self, student: StudentProfile) -> Dict[str, Any]:
        """Create a summary of the student profile used in recommendations.
        
        Args:
            student: Student profile
            
        Returns:
            Dictionary with student profile summary
        """
        return {
            "gpa": student.gpa,
            "budget_max": student.budget_max,
            "field_of_study": student.field_of_study,
            "preferred_countries": student.preferred_countries,
            "preferred_cities": student.preferred_cities,
            "test_scores": student.test_scores,
            "priorities": student.priorities,
            "university_type_preference": student.university_type_preference.value if student.university_type_preference else None,
            "campus_culture_preferences": [c.value for c in student.campus_culture_preferences]
        }
    
    def get_university_by_id(self, university_id: str) -> Optional[University]:
        """Get a specific university by ID.
        
        Args:
            university_id: University identifier
            
        Returns:
            University object or None if not found
        """
        return self.data_processor.get_university_by_id(university_id)
    
    def get_all_universities(self) -> List[University]:
        """Get all universities in the dataset.
        
        Returns:
            List of all universities
        """
        return self.data_processor.get_universities()
    
    def get_filtered_universities(self, filter_criteria: FilterCriteria) -> List[University]:
        """Get universities filtered by specific criteria.
        
        Args:
            filter_criteria: Filtering criteria
            
        Returns:
            Filtered list of universities
        """
        universities = self.data_processor.get_universities()
        
        if filter_criteria.countries:
            universities = [u for u in universities if u.country in filter_criteria.countries]
        
        if filter_criteria.max_tuition is not None:
            universities = [u for u in universities if u.tuition_fees_usd <= filter_criteria.max_tuition]
        
        if filter_criteria.max_total_cost is not None:
            universities = [u for u in universities 
                          if (u.tuition_fees_usd + u.living_cost_usd) <= filter_criteria.max_total_cost]
        
        if filter_criteria.min_gpa_requirement is not None:
            universities = [u for u in universities if u.min_gpa <= filter_criteria.min_gpa_requirement]
        
        if filter_criteria.programs:
            universities = [u for u in universities 
                          if any(prog.lower() in [p.lower() for p in u.programs_offered] 
                                for prog in filter_criteria.programs)]
        
        if filter_criteria.university_types:
            universities = [u for u in universities if u.university_type in filter_criteria.university_types]
        
        if filter_criteria.min_acceptance_rate is not None:
            universities = [u for u in universities if u.acceptance_rate >= filter_criteria.min_acceptance_rate]
        
        if filter_criteria.max_acceptance_rate is not None:
            universities = [u for u in universities if u.acceptance_rate <= filter_criteria.max_acceptance_rate]
        
        return universities
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the university dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        return self.data_processor.get_statistics()
    
    def refresh_data(self) -> bool:
        """Refresh the university dataset and retrain models.
        
        Returns:
            True if refresh was successful
        """
        try:
            # Reload data
            self.data_processor._load_data()
            
            # Retrain ML models
            universities = self.data_processor.get_universities()
            if universities:
                self.ml_recommender.train(universities)
                logger.info("Data refresh completed successfully")
                return True
            else:
                logger.error("No universities found after data refresh")
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return False
    
    def validate_student_profile(self, student: StudentProfile) -> List[str]:
        """Validate student profile and return any warnings or errors.
        
        Args:
            student: Student profile to validate
            
        Returns:
            List of validation messages
        """
        warnings = []
        
        # Check GPA range
        if student.gpa < 0.0 or student.gpa > 4.0:
            warnings.append("GPA should be between 0.0 and 4.0")
        
        # Check budget reasonableness
        if student.budget_max < 5000:
            warnings.append("Budget seems very low for international education")
        elif student.budget_max > 200000:
            warnings.append("Budget seems unusually high")
        
        # Check if field of study is specified
        if not student.field_of_study.strip():
            warnings.append("Field of study should be specified for better recommendations")
        
        # Check priorities sum
        if student.priorities:
            priority_sum = sum(student.priorities.values())
            if abs(priority_sum - 1.0) > 0.1:
                warnings.append(f"Priority weights should sum to 1.0 (current sum: {priority_sum:.2f})")
        
        # Check test scores reasonableness
        for test, score in student.test_scores.items():
            if test.upper() == 'SAT' and (score < 400 or score > 1600):
                warnings.append("SAT score should be between 400 and 1600")
            elif test.upper() == 'TOEFL' and (score < 0 or score > 120):
                warnings.append("TOEFL score should be between 0 and 120")
            elif test.upper() == 'IELTS' and (score < 0 or score > 9):
                warnings.append("IELTS score should be between 0 and 9")
        
        return warnings 