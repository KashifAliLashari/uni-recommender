"""
Unit tests for the University Matching Algorithm.
"""
import pytest
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import StudentProfile, University, UniversityType, CampusCulture
from src.matching_algorithm import UniversityMatchingAlgorithm
from src.data_processor import UniversityDataProcessor
from src.recommendation_engine import UniversityRecommendationEngine


class TestMatchingAlgorithm:
    """Test cases for the matching algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.algorithm = UniversityMatchingAlgorithm()
        
        # Create sample universities
        self.universities = [
            University(
                university_id="test_1",
                name="Test University 1",
                country="USA",
                city="Boston",
                ranking_global=50,
                ranking_national=10,
                tuition_fees_usd=50000,
                living_cost_usd=15000,
                min_gpa=3.5,
                min_test_scores={"SAT": 1400},
                acceptance_rate=0.3,
                programs_offered=["Computer Science", "Engineering"],
                languages_taught=["English"],
                university_type=UniversityType.PRIVATE,
                reputation_score=85.0
            ),
            University(
                university_id="test_2",
                name="Test University 2",
                country="Canada",
                city="Toronto",
                ranking_global=100,
                ranking_national=5,
                tuition_fees_usd=30000,
                living_cost_usd=12000,
                min_gpa=3.0,
                min_test_scores={"SAT": 1200},
                acceptance_rate=0.5,
                programs_offered=["Business", "Engineering"],
                languages_taught=["English"],
                university_type=UniversityType.PUBLIC,
                reputation_score=75.0
            )
        ]
        
        # Create sample student
        self.student = StudentProfile(
            gpa=3.7,
            test_scores={"SAT": 1450},
            budget_max=70000,
            preferred_countries=["USA", "Canada"],
            field_of_study="Computer Science",
            timeline="Fall 2024",
            university_size_preference="Medium",
            priorities={
                "academic_reputation": 0.3,
                "cost": 0.25,
                "location": 0.2,
                "culture": 0.15,
                "admission_probability": 0.1
            }
        )
    
    def test_hard_filters_budget(self):
        """Test that budget filtering works correctly."""
        # Create a student with low budget
        low_budget_student = StudentProfile(
            gpa=3.7,
            test_scores={"SAT": 1450},
            budget_max=35000,  # Lower than Test University 1 total cost
            preferred_countries=["USA", "Canada"],
            field_of_study="Computer Science",
            timeline="Fall 2024",
            university_size_preference="Medium"
        )
        
        filtered = self.algorithm.apply_hard_filters(self.universities, low_budget_student)
        
        # Should filter out expensive universities
        # Both test universities are over 35000 budget (65000 and 42000)
        assert len(filtered) == 0
    
    def test_hard_filters_gpa(self):
        """Test that GPA filtering works correctly."""
        # Create a student with low GPA
        low_gpa_student = StudentProfile(
            gpa=2.8,  # Lower than minimum requirements
            test_scores={"SAT": 1450},
            budget_max=100000,
            preferred_countries=["USA", "Canada"],
            field_of_study="Computer Science",
            timeline="Fall 2024",
            university_size_preference="Medium"
        )
        
        filtered = self.algorithm.apply_hard_filters(self.universities, low_gpa_student)
        
        # Should be empty since GPA is too low for both universities
        assert len(filtered) == 0
    
    def test_hard_filters_country(self):
        """Test that country filtering works correctly."""
        # Create a student with specific country preference
        country_specific_student = StudentProfile(
            gpa=3.7,
            test_scores={"SAT": 1450},
            budget_max=100000,
            preferred_countries=["USA"],  # Only USA
            field_of_study="Computer Science",
            timeline="Fall 2024",
            university_size_preference="Medium"
        )
        
        filtered = self.algorithm.apply_hard_filters(self.universities, country_specific_student)
        
        # Should only include Test University 1 (USA)
        assert len(filtered) == 1
        assert filtered[0].country == "USA"
    
    def test_scoring_academic_fit(self):
        """Test academic fit scoring."""
        university = self.universities[0]  # High reputation university
        score = self.algorithm._calculate_academic_fit_score(university, self.student)
        
        assert 0 <= score <= 100
        assert score > 60  # Should be reasonably high for good reputation university
    
    def test_scoring_financial_feasibility(self):
        """Test financial feasibility scoring."""
        university = self.universities[1]  # Lower cost university
        score = self.algorithm._calculate_financial_feasibility_score(university, self.student)
        
        assert 0 <= score <= 100
        # Total cost (42000) is well within budget (70000), so should score high
        assert score >= 80
    
    def test_scoring_admission_probability(self):
        """Test admission probability scoring."""
        university = self.universities[1]  # Higher acceptance rate
        score = self.algorithm._calculate_admission_probability_score(university, self.student)
        
        assert 0 <= score <= 100
        # Student exceeds requirements, so should have good probability
        assert score > 50
    
    def test_complete_matching_process(self):
        """Test the complete matching process."""
        filtered = self.algorithm.apply_hard_filters(self.universities, self.student)
        scored = self.algorithm.calculate_match_scores(filtered, self.student)
        
        assert len(scored) > 0
        
        # Check that scores are properly calculated
        for university, match_score in scored:
            assert 0 <= match_score.academic_fit <= 100
            assert 0 <= match_score.financial_feasibility <= 100
            assert 0 <= match_score.preference_alignment <= 100
            assert 0 <= match_score.admission_probability <= 100
            assert 0 <= match_score.overall_score <= 100
        
        # Check that results are sorted by overall score
        scores = [match_score.overall_score for _, match_score in scored]
        assert scores == sorted(scores, reverse=True)
    
    def test_explanation_generation(self):
        """Test explanation generation."""
        university = self.universities[0]
        
        # Create a simple match score
        from src.models import MatchScore
        match_score = MatchScore(
            academic_fit=85.0,
            financial_feasibility=70.0,
            preference_alignment=80.0,
            admission_probability=75.0,
            overall_score=78.0
        )
        
        explanation = self.algorithm.generate_explanation(university, match_score, self.student)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        # Should mention key factors
        assert "academic" in explanation.lower() or "reputation" in explanation.lower()
    
    def test_concerns_identification(self):
        """Test identification of concerns."""
        # Test with expensive university
        expensive_university = University(
            university_id="expensive",
            name="Expensive University",
            country="USA",
            city="New York",
            ranking_global=5,
            ranking_national=2,
            tuition_fees_usd=65000,
            living_cost_usd=20000,
            min_gpa=3.8,
            acceptance_rate=0.05,  # Very competitive
            programs_offered=["Computer Science"],
            languages_taught=["English"],
            university_type=UniversityType.PRIVATE,
            reputation_score=95.0
        )
        
        concerns = self.algorithm.identify_concerns(expensive_university, self.student)
        
        assert isinstance(concerns, list)
        # Should identify competitive admission as a concern
        assert any("competitive" in concern.lower() for concern in concerns)


class TestDataProcessor:
    """Test cases for the data processor."""
    
    def test_sample_data_creation(self):
        """Test that sample data is created correctly."""
        processor = UniversityDataProcessor("test_nonexistent.csv")
        
        universities = processor.get_universities()
        assert len(universities) > 0
        
        # Check that universities have required fields
        for university in universities[:3]:  # Test first 3
            assert university.name
            assert university.country
            assert university.city
            assert university.tuition_fees_usd >= 0
            assert university.living_cost_usd >= 0
            assert 0 <= university.min_gpa <= 4.0
            assert 0 <= university.acceptance_rate <= 1.0
    
    def test_filtering(self):
        """Test university filtering."""
        processor = UniversityDataProcessor("test_nonexistent.csv")
        
        # Test country filtering
        us_universities = processor.filter_universities(countries=["USA"])
        assert all(u.country == "USA" for u in us_universities)
        
        # Test budget filtering
        affordable_universities = processor.filter_universities(max_tuition=40000)
        assert all(u.tuition_fees_usd <= 40000 for u in affordable_universities)
    
    def test_statistics(self):
        """Test statistics generation."""
        processor = UniversityDataProcessor("test_nonexistent.csv")
        stats = processor.get_statistics()
        
        assert "total_universities" in stats
        assert "countries" in stats
        assert "avg_tuition" in stats
        assert stats["total_universities"] > 0


class TestRecommendationEngine:
    """Test cases for the recommendation engine."""
    
    def test_engine_initialization(self):
        """Test that the recommendation engine initializes correctly."""
        engine = UniversityRecommendationEngine("test_nonexistent.csv")
        
        assert engine.is_initialized
        assert len(engine.get_all_universities()) > 0
    
    def test_recommendation_generation(self):
        """Test recommendation generation."""
        engine = UniversityRecommendationEngine("test_nonexistent.csv")
        
        student = StudentProfile(
            gpa=3.7,
            test_scores={"SAT": 1450},
            budget_max=60000,
            preferred_countries=["USA"],
            field_of_study="Computer Science",
            timeline="Fall 2024",
            university_size_preference="Medium"
        )
        
        response = engine.get_recommendations(student, max_recommendations=5)
        
        assert response.total_universities_considered > 0
        assert len(response.recommendations) <= 5
        assert response.processing_time_ms > 0
        
        # Check recommendation structure
        for rec in response.recommendations:
            assert rec.university.name
            assert 0 <= rec.match_score.overall_score <= 100
            assert rec.total_annual_cost > 0
            assert rec.rank_position > 0
    
    def test_student_profile_validation(self):
        """Test student profile validation."""
        engine = UniversityRecommendationEngine("test_nonexistent.csv")
        
        # Valid profile
        valid_student = StudentProfile(
            gpa=3.7,
            test_scores={"SAT": 1450},
            budget_max=60000,
            preferred_countries=["USA"],
            field_of_study="Computer Science",
            timeline="Fall 2024",
            university_size_preference="Medium"
        )
        
        warnings = engine.validate_student_profile(valid_student)
        assert isinstance(warnings, list)
        
        # Profile with validation issues that pass Pydantic but trigger warnings
        problematic_student = StudentProfile(
            gpa=2.0,  # Valid but low GPA
            test_scores={"SAT": 800},  # Valid but low SAT score
            budget_max=5000,  # Very low budget
            preferred_countries=["USA"],
            field_of_study="",  # Empty field
            timeline="Fall 2024",
            university_size_preference="Medium"
        )
        
        warnings = engine.validate_student_profile(problematic_student)
        assert len(warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 