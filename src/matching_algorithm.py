"""
Core matching algorithm for university recommendations.
Implements rule-based filtering and multi-dimensional scoring.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from .models import StudentProfile, University, MatchScore, FilterCriteria

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for different scoring components."""
    academic_reputation: float = 0.3
    financial_feasibility: float = 0.25
    location_preference: float = 0.2
    culture_fit: float = 0.15
    admission_probability: float = 0.1


class UniversityMatchingAlgorithm:
    """Core matching algorithm for university recommendations."""
    
    def __init__(self):
        """Initialize the matching algorithm."""
        self.scoring_weights = ScoringWeights()
    
    def apply_hard_filters(self, 
                          universities: List[University], 
                          student: StudentProfile) -> List[University]:
        """Apply hard constraint filters to universities.
        
        Args:
            universities: List of all universities
            student: Student profile with constraints
            
        Returns:
            Filtered list of universities meeting hard constraints
        """
        filtered = universities.copy()
        
        # Budget constraint (tuition + living costs)
        filtered = [u for u in filtered 
                   if (u.tuition_fees_usd + u.living_cost_usd) <= student.budget_max]
        
        # GPA requirement
        filtered = [u for u in filtered if u.min_gpa <= student.gpa]
        
        # Country preference (if specified)
        if student.preferred_countries:
            filtered = [u for u in filtered if u.country in student.preferred_countries]
        
        # City preference (if specified)
        if student.preferred_cities:
            filtered = [u for u in filtered if u.city in student.preferred_cities]
        
        # Field of study requirement
        if student.field_of_study:
            filtered = [u for u in filtered 
                       if self._has_program(u, student.field_of_study)]
        
        # Language requirements
        if student.language_requirements:
            filtered = [u for u in filtered 
                       if self._meets_language_requirements(u, student.language_requirements)]
        
        # Test score requirements
        if student.test_scores:
            filtered = [u for u in filtered 
                       if self._meets_test_score_requirements(u, student.test_scores)]
        
        logger.info(f"Applied hard filters: {len(universities)} -> {len(filtered)} universities")
        return filtered
    
    def _has_program(self, university: University, field_of_study: str) -> bool:
        """Check if university offers the required field of study.
        
        Args:
            university: University to check
            field_of_study: Required field of study
            
        Returns:
            True if university offers the program
        """
        if not university.programs_offered:
            return True  # Assume all programs available if not specified
        
        field_lower = field_of_study.lower()
        return any(field_lower in program.lower() for program in university.programs_offered)
    
    def _meets_language_requirements(self, 
                                   university: University, 
                                   language_requirements: Dict[str, str]) -> bool:
        """Check if university meets language requirements.
        
        Args:
            university: University to check
            language_requirements: Student's language proficiency
            
        Returns:
            True if requirements are met
        """
        # Simplified language checking - in practice, would need more sophisticated logic
        if not university.languages_taught:
            return True
        
        # Check if student has proficiency in any of the taught languages
        taught_languages = [lang.lower() for lang in university.languages_taught]
        student_languages = [lang.lower() for lang in language_requirements.keys()]
        
        return any(lang in taught_languages for lang in student_languages) or 'english' in taught_languages
    
    def _meets_test_score_requirements(self, 
                                     university: University, 
                                     student_scores: Dict[str, int]) -> bool:
        """Check if student meets test score requirements.
        
        Args:
            university: University to check
            student_scores: Student's test scores
            
        Returns:
            True if test score requirements are met
        """
        if not university.min_test_scores:
            return True
        
        for test, min_score in university.min_test_scores.items():
            if test in student_scores:
                if student_scores[test] < min_score:
                    return False
        
        return True
    
    def calculate_match_scores(self, 
                             universities: List[University], 
                             student: StudentProfile) -> List[Tuple[University, MatchScore]]:
        """Calculate match scores for filtered universities.
        
        Args:
            universities: Filtered list of universities
            student: Student profile
            
        Returns:
            List of (University, MatchScore) tuples
        """
        scored_matches = []
        
        for university in universities:
            # Calculate individual scores
            academic_score = self._calculate_academic_fit_score(university, student)
            financial_score = self._calculate_financial_feasibility_score(university, student)
            preference_score = self._calculate_preference_alignment_score(university, student)
            admission_score = self._calculate_admission_probability_score(university, student)
            
            # Calculate weighted overall score
            weights = student.priorities
            overall_score = (
                academic_score * weights.get('academic_reputation', 0.3) +
                financial_score * weights.get('cost', 0.25) +
                preference_score * weights.get('location', 0.2) +
                preference_score * weights.get('culture', 0.15) +  # Using preference for culture
                admission_score * weights.get('admission_probability', 0.1)
            )
            
            match_score = MatchScore(
                academic_fit=academic_score,
                financial_feasibility=financial_score,
                preference_alignment=preference_score,
                admission_probability=admission_score,
                overall_score=overall_score
            )
            
            scored_matches.append((university, match_score))
        
        # Sort by overall score (descending)
        scored_matches.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return scored_matches
    
    def _calculate_academic_fit_score(self, university: University, student: StudentProfile) -> float:
        """Calculate academic fit score (0-100).
        
        Args:
            university: University to score
            student: Student profile
            
        Returns:
            Academic fit score
        """
        score = 0.0
        
        # Reputation component (40% of academic score)
        reputation_score = min(university.reputation_score, 100.0)
        score += reputation_score * 0.4
        
        # Ranking component (30% of academic score)
        if university.ranking_global:
            # Convert ranking to score (lower rank = higher score)
            ranking_score = max(0, 100 - (university.ranking_global / 100))
            score += ranking_score * 0.3
        else:
            score += 50 * 0.3  # Default score for unranked universities
        
        # GPA fit component (20% of academic score)
        gpa_difference = student.gpa - university.min_gpa
        gpa_score = min(100, max(0, 60 + (gpa_difference * 20)))  # Base 60, bonus for exceeding
        score += gpa_score * 0.2
        
        # Research opportunities (10% of academic score)
        research_score = 80 if university.research_opportunities else 40
        score += research_score * 0.1
        
        return min(100.0, score)
    
    def _calculate_financial_feasibility_score(self, university: University, student: StudentProfile) -> float:
        """Calculate financial feasibility score (0-100).
        
        Args:
            university: University to score
            student: Student profile
            
        Returns:
            Financial feasibility score
        """
        total_cost = university.tuition_fees_usd + university.living_cost_usd
        budget_ratio = total_cost / student.budget_max
        
        if budget_ratio <= 0.7:
            return 100.0  # Very affordable
        elif budget_ratio <= 0.85:
            return 80.0   # Affordable
        elif budget_ratio <= 1.0:
            return 60.0   # Within budget but tight
        else:
            return 0.0    # Over budget (should be filtered out)
    
    def _calculate_preference_alignment_score(self, university: University, student: StudentProfile) -> float:
        """Calculate preference alignment score (0-100).
        
        Args:
            university: University to score
            student: Student profile
            
        Returns:
            Preference alignment score
        """
        score = 0.0
        
        # Country preference (40% of preference score)
        if student.preferred_countries:
            country_score = 100 if university.country in student.preferred_countries else 20
        else:
            country_score = 70  # No preference specified
        score += country_score * 0.4
        
        # City preference (20% of preference score)
        if student.preferred_cities:
            city_score = 100 if university.city in student.preferred_cities else 30
        else:
            city_score = 70  # No preference specified
        score += city_score * 0.2
        
        # University type preference (20% of preference score)
        if student.university_type_preference:
            type_score = 100 if university.university_type == student.university_type_preference else 40
        else:
            type_score = 70
        score += type_score * 0.2
        
        # Campus culture alignment (20% of preference score)
        if student.campus_culture_preferences:
            culture_matches = len(set(student.campus_culture_preferences) & 
                                set(university.campus_culture_tags))
            culture_score = min(100, (culture_matches / len(student.campus_culture_preferences)) * 100)
        else:
            culture_score = 70
        score += culture_score * 0.2
        
        return min(100.0, score)
    
    def _calculate_admission_probability_score(self, university: University, student: StudentProfile) -> float:
        """Calculate admission probability score (0-100).
        
        Args:
            university: University to score
            student: Student profile
            
        Returns:
            Admission probability score
        """
        # Base probability from acceptance rate
        base_probability = university.acceptance_rate * 100
        
        # Adjust based on student's academic profile
        gpa_bonus = max(0, (student.gpa - university.min_gpa) * 10)
        
        # Test score bonus
        test_bonus = 0
        if student.test_scores and university.min_test_scores:
            for test, student_score in student.test_scores.items():
                if test in university.min_test_scores:
                    min_required = university.min_test_scores[test]
                    if student_score > min_required:
                        test_bonus += min(10, (student_score - min_required) / min_required * 10)
        
        # International student factor
        if university.international_student_ratio > 0.3:
            international_bonus = 10  # Universities with high international ratios are more welcoming
        else:
            international_bonus = 0
        
        final_probability = min(100, base_probability + gpa_bonus + test_bonus + international_bonus)
        return final_probability
    
    def generate_explanation(self, university: University, match_score: MatchScore, student: StudentProfile) -> str:
        """Generate explanation for why a university was recommended.
        
        Args:
            university: Recommended university
            match_score: Calculated match scores
            student: Student profile
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Academic fit explanation
        if match_score.academic_fit > 80:
            explanations.append(f"Excellent academic fit with strong reputation (score: {university.reputation_score:.1f})")
        elif match_score.academic_fit > 60:
            explanations.append(f"Good academic match with solid reputation")
        
        # Financial explanation
        total_cost = university.tuition_fees_usd + university.living_cost_usd
        if match_score.financial_feasibility > 80:
            explanations.append(f"Very affordable at ${total_cost:,.0f} annually (well within your ${student.budget_max:,.0f} budget)")
        elif match_score.financial_feasibility > 60:
            explanations.append(f"Within budget at ${total_cost:,.0f} annually")
        
        # Location explanation
        if university.country in student.preferred_countries:
            explanations.append(f"Located in your preferred country ({university.country})")
        
        # Program explanation
        if self._has_program(university, student.field_of_study):
            explanations.append(f"Offers strong programs in {student.field_of_study}")
        
        # Admission probability
        if match_score.admission_probability > 70:
            explanations.append(f"High admission probability ({match_score.admission_probability:.0f}%)")
        elif match_score.admission_probability > 50:
            explanations.append(f"Reasonable admission chances ({match_score.admission_probability:.0f}%)")
        
        return " • ".join(explanations) if explanations else "Good overall match based on your profile"
    
    def identify_concerns(self, university: University, student: StudentProfile) -> List[str]:
        """Identify potential concerns or challenges.
        
        Args:
            university: University to analyze
            student: Student profile
            
        Returns:
            List of concern strings
        """
        concerns = []
        
        # Financial concerns
        total_cost = university.tuition_fees_usd + university.living_cost_usd
        budget_ratio = total_cost / student.budget_max
        if budget_ratio > 0.9:
            concerns.append(f"High cost (${total_cost:,.0f}) uses {budget_ratio*100:.0f}% of your budget")
        
        # Competitive admission
        if university.acceptance_rate < 0.15:
            concerns.append(f"Highly competitive admission (only {university.acceptance_rate*100:.1f}% acceptance rate)")
        
        # GPA close to minimum
        gpa_margin = student.gpa - university.min_gpa
        if gpa_margin < 0.2:
            concerns.append(f"GPA requirement is close to your current GPA (requires {university.min_gpa:.1f})")
        
        # Language requirements
        if 'English' not in university.languages_taught and 'english' not in [lang.lower() for lang in student.language_requirements.keys()]:
            primary_language = university.languages_taught[0] if university.languages_taught else "local language"
            concerns.append(f"Instruction primarily in {primary_language}")
        
        return concerns 