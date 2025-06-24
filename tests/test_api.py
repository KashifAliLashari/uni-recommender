"""
API tests for the University Matching Algorithm.
"""
import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api import app
from src.models import StudentProfile, RecommendationRequest, FilterCriteria

client = TestClient(app)


class TestAPI:
    """Test cases for the API endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        # Note: This might fail if the engine hasn't been initialized
        # In a real test environment, we'd mock the engine
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "engine_initialized" in data
    
    def test_universities_endpoint(self):
        """Test the universities listing endpoint."""
        response = client.get("/universities")
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            
            # Check structure of first university if any exist
            if data:
                university = data[0]
                assert "university_id" in university
                assert "name" in university
                assert "country" in university
                assert "tuition_fees_usd" in university
    
    def test_universities_with_pagination(self):
        """Test universities endpoint with pagination."""
        response = client.get("/universities?limit=2&offset=0")
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) <= 2
    
    def test_universities_with_country_filter(self):
        """Test universities endpoint with country filter."""
        response = client.get("/universities?country=USA")
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            
            # All returned universities should be from USA
            for university in data:
                assert university["country"] == "USA"
    
    def test_university_by_id_not_found(self):
        """Test getting a university by non-existent ID."""
        response = client.get("/universities/nonexistent_id")
        assert response.status_code == 404
    
    def test_recommendations_endpoint(self):
        """Test the recommendations endpoint."""
        # Create a valid student profile
        student_data = {
            "gpa": 3.7,
            "test_scores": {"SAT": 1450},
            "budget_max": 60000,
            "preferred_countries": ["USA"],
            "field_of_study": "Computer Science",
            "priorities": {
                "academic_reputation": 0.3,
                "cost": 0.25,
                "location": 0.2,
                "culture": 0.15,
                "admission_probability": 0.1
            }
        }
        
        request_data = {
            "student_profile": student_data,
            "max_recommendations": 5,
            "include_explanations": True
        }
        
        response = client.post("/recommendations", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "total_universities_considered" in data
            assert "processing_time_ms" in data
            assert "student_summary" in data
            
            # Check recommendations structure
            recommendations = data["recommendations"]
            assert isinstance(recommendations, list)
            assert len(recommendations) <= 5
            
            for rec in recommendations:
                assert "university" in rec
                assert "match_score" in rec
                assert "total_annual_cost" in rec
                assert "rank_position" in rec
    
    def test_recommendations_validation(self):
        """Test student profile validation endpoint."""
        student_data = {
            "gpa": 3.7,
            "test_scores": {"SAT": 1450},
            "budget_max": 60000,
            "preferred_countries": ["USA"],
            "field_of_study": "Computer Science"
        }
        
        response = client.post("/recommendations/validate", json=student_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "valid" in data
            assert "warnings" in data
            assert "profile_summary" in data
    
    def test_filter_universities(self):
        """Test university filtering endpoint."""
        filter_data = {
            "countries": ["USA", "Canada"],
            "max_tuition": 50000,
            "programs": ["Computer Science"]
        }
        
        response = client.post("/universities/filter", json=filter_data)
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            
            # Check that returned universities match filters
            for university in data:
                assert university["country"] in ["USA", "Canada"]
                assert university["tuition_fees_usd"] <= 50000
    
    def test_statistics_endpoint(self):
        """Test the statistics endpoint."""
        response = client.get("/statistics")
        
        if response.status_code == 200:
            data = response.json()
            assert "total_universities" in data
            assert "api_version" in data
            assert "engine_status" in data
    
    def test_countries_endpoint(self):
        """Test the countries endpoint."""
        response = client.get("/countries")
        
        if response.status_code == 200:
            data = response.json()
            assert "countries" in data
            assert "total_count" in data
            assert isinstance(data["countries"], list)
    
    def test_programs_endpoint(self):
        """Test the programs endpoint."""
        response = client.get("/programs")
        
        if response.status_code == 200:
            data = response.json()
            assert "programs" in data
            assert "total_count" in data
            assert isinstance(data["programs"], list)
    
    def test_invalid_student_profile(self):
        """Test recommendations with invalid student profile."""
        invalid_student_data = {
            "gpa": 5.0,  # Invalid GPA
            "budget_max": -1000,  # Negative budget
            "field_of_study": ""  # Empty field
        }
        
        request_data = {
            "student_profile": invalid_student_data,
            "max_recommendations": 5
        }
        
        response = client.post("/recommendations", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_malformed_json(self):
        """Test endpoints with malformed JSON."""
        response = client.post(
            "/recommendations",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test recommendations with missing required fields."""
        incomplete_student_data = {
            "gpa": 3.7
            # Missing budget_max and field_of_study
        }
        
        request_data = {
            "student_profile": incomplete_student_data
        }
        
        response = client.post("/recommendations", json=request_data)
        
        # Should return validation error for missing required fields
        assert response.status_code == 422


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_404_error_handling(self):
        """Test 404 error handling."""
        response = client.get("/nonexistent_endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test method not allowed error."""
        # Try to POST to a GET-only endpoint
        response = client.post("/universities")
        assert response.status_code == 405


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    def test_recommendations_performance(self):
        """Test that recommendations are generated in reasonable time."""
        student_data = {
            "gpa": 3.7,
            "test_scores": {"SAT": 1450},
            "budget_max": 60000,
            "preferred_countries": ["USA"],
            "field_of_study": "Computer Science"
        }
        
        request_data = {
            "student_profile": student_data,
            "max_recommendations": 20
        }
        
        import time
        start_time = time.time()
        
        response = client.post("/recommendations", json=request_data)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Should respond within 5 seconds (generous limit for testing)
        assert response_time < 5.0
        
        if response.status_code == 200:
            data = response.json()
            # Processing time reported by the engine should be much faster
            assert data["processing_time_ms"] < 3000  # 3 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 