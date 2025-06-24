"""
Example usage of the University Matching Algorithm.
This script demonstrates how to use the recommendation engine to get university recommendations.
"""
import json
import logging
from src.recommendation_engine import UniversityRecommendationEngine
from src.models import StudentProfile, UniversityType, CampusCulture

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_students():
    """Load sample students from JSON file."""
    try:
        with open('data/sample_students.json', 'r') as f:
            sample_data = json.load(f)
        return sample_data
    except FileNotFoundError:
        logger.warning("Sample students file not found. Using hardcoded examples.")
        return []


def create_example_student():
    """Create an example student profile."""
    return StudentProfile(
        gpa=3.7,
        test_scores={"SAT": 1450, "TOEFL": 95},
        qualifications=["AP Computer Science", "Math Competition"],
        preferred_countries=["USA", "Canada"],
        preferred_cities=["Boston", "Toronto"],
        budget_max=60000,
        field_of_study="Computer Science",
        language_requirements={"English": "Advanced"},
        timeline="Fall 2024",
        priorities={
            "academic_reputation": 0.35,
            "cost": 0.25,
            "location": 0.2,
            "culture": 0.15,
            "admission_probability": 0.05
        },
        university_size_preference="Medium",
        university_type_preference=UniversityType.PRIVATE,
        campus_culture_preferences=[CampusCulture.RESEARCH_FOCUSED, CampusCulture.TECHNICAL]
    )


def print_recommendations(response):
    """Print recommendations in a formatted way."""
    print(f"\n{'='*80}")
    print(f"UNIVERSITY RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"Total universities considered: {response.total_universities_considered}")
    print(f"Universities after filtering: {response.universities_after_filtering}")
    print(f"Processing time: {response.processing_time_ms:.2f} ms")
    print(f"Number of recommendations: {len(response.recommendations)}")
    
    print(f"\n{'='*80}")
    print(f"TOP RECOMMENDATIONS")
    print(f"{'='*80}")
    
    for i, rec in enumerate(response.recommendations[:5], 1):
        print(f"\n{i}. {rec.university.name}")
        print(f"   Location: {rec.university.city}, {rec.university.country}")
        print(f"   Overall Score: {rec.match_score.overall_score:.1f}/100")
        print(f"   Total Annual Cost: ${rec.total_annual_cost:,.0f}")
        
        print(f"   Detailed Scores:")
        print(f"     • Academic Fit: {rec.match_score.academic_fit:.1f}/100")
        print(f"     • Financial Feasibility: {rec.match_score.financial_feasibility:.1f}/100")
        print(f"     • Preference Alignment: {rec.match_score.preference_alignment:.1f}/100")
        print(f"     • Admission Probability: {rec.match_score.admission_probability:.1f}/100")
        
        if rec.explanation:
            print(f"   Why recommended: {rec.explanation}")
        
        if rec.concerns:
            print(f"   Potential concerns:")
            for concern in rec.concerns:
                print(f"     • {concern}")


def demonstrate_api_features(engine):
    """Demonstrate various API features."""
    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS")
    print(f"{'='*80}")
    
    stats = engine.get_dataset_statistics()
    print(f"Total universities: {stats.get('total_universities', 0)}")
    print(f"Countries available: {', '.join(stats.get('countries', []))}")
    print(f"Average tuition: ${stats.get('avg_tuition', 0):,.0f}")
    print(f"Average minimum GPA: {stats.get('avg_min_gpa', 0):.2f}")
    
    print(f"\nPrograms available: {', '.join(stats.get('programs_available', [])[:10])}...")


def run_example():
    """Run the complete example."""
    print("University Matching Algorithm - Example Usage")
    print("=" * 80)
    
    # Initialize the recommendation engine
    print("Initializing recommendation engine...")
    engine = UniversityRecommendationEngine()
    
    if not engine.is_initialized:
        print("ERROR: Could not initialize recommendation engine!")
        return
    
    print("✓ Recommendation engine initialized successfully")
    
    # Show dataset statistics
    demonstrate_api_features(engine)
    
    # Create example student
    print(f"\n{'='*80}")
    print(f"CREATING STUDENT PROFILE")
    print(f"{'='*80}")
    
    student = create_example_student()
    print(f"Student Profile:")
    print(f"  GPA: {student.gpa}")
    print(f"  Test Scores: {student.test_scores}")
    print(f"  Budget: ${student.budget_max:,.0f}")
    print(f"  Field of Study: {student.field_of_study}")
    print(f"  Preferred Countries: {', '.join(student.preferred_countries)}")
    print(f"  University Type Preference: {student.university_type_preference}")
    
    # Validate student profile
    warnings = engine.validate_student_profile(student)
    if warnings:
        print(f"\nProfile Validation Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
    else:
        print("\n✓ Student profile is valid")
    
    # Get recommendations
    print(f"\n{'='*80}")
    print(f"GENERATING RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print("Generating recommendations...")
    response = engine.get_recommendations(
        student_profile=student,
        max_recommendations=10,
        include_explanations=True
    )
    
    # Print results
    print_recommendations(response)
    
    # Try with a different student (budget-conscious)
    print(f"\n{'='*80}")
    print(f"BUDGET-CONSCIOUS STUDENT EXAMPLE")
    print(f"{'='*80}")
    
    budget_student = StudentProfile(
        gpa=3.4,
        test_scores={"SAT": 1300, "TOEFL": 85},
        budget_max=35000,  # Lower budget
        preferred_countries=["Canada", "Germany"],
        field_of_study="Engineering",
        timeline="Fall 2024",
        university_size_preference="Medium",
        priorities={
            "academic_reputation": 0.2,
            "cost": 0.5,  # Higher priority on cost
            "location": 0.2,
            "culture": 0.05,
            "admission_probability": 0.05
        }
    )
    
    print("Generating recommendations for budget-conscious student...")
    budget_response = engine.get_recommendations(
        student_profile=budget_student,
        max_recommendations=5,
        include_explanations=True
    )
    
    print_recommendations(budget_response)
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        run_example()
    except Exception as e:
        logger.error(f"Error running example: {e}")
        import traceback
        traceback.print_exc() 