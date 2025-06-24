"""
University data processing module.
Handles loading, cleaning, and preprocessing of university dataset.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import logging
from pathlib import Path

from .models import University, UniversityType, CampusCulture

logger = logging.getLogger(__name__)


class UniversityDataProcessor:
    """Processes and manages university dataset."""
    
    def __init__(self, data_file_path: str = "data/universities.csv"):
        """Initialize the data processor.
        
        Args:
            data_file_path: Path to the university dataset CSV file
        """
        self.data_file_path = data_file_path
        self.universities_df: Optional[pd.DataFrame] = None
        self.universities: List[University] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load university data from CSV file."""
        try:
            if Path(self.data_file_path).exists():
                self.universities_df = pd.read_csv(self.data_file_path)
                logger.info(f"Loaded {len(self.universities_df)} universities from {self.data_file_path}")
                self._preprocess_data()
                self._convert_to_models()
            else:
                logger.warning(f"Data file {self.data_file_path} not found. Creating sample data.")
                self._create_sample_data()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._create_sample_data()
    
    def _preprocess_data(self) -> None:
        """Clean and preprocess the university data."""
        if self.universities_df is None:
            return
        
        # Handle missing values
        self.universities_df = self.universities_df.fillna({
            'ranking_global': 9999,
            'ranking_national': 9999,
            'reputation_score': 50.0,
            'acceptance_rate': 0.5,
            'international_student_ratio': 0.1,
            'student_population': 10000,
            'has_scholarship': False,
            'research_opportunities': True
        })
        
        # Convert string columns to lists where needed
        list_columns = ['programs_offered', 'languages_taught', 'campus_culture_tags']
        for col in list_columns:
            if col in self.universities_df.columns:
                self.universities_df[col] = self.universities_df[col].apply(
                    lambda x: x.split(';') if isinstance(x, str) else (x if isinstance(x, list) else [])
                )
        
        # Parse test score requirements
        if 'min_test_scores' in self.universities_df.columns:
            self.universities_df['min_test_scores'] = self.universities_df['min_test_scores'].apply(
                lambda x: self._parse_test_scores(x)
            )
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            'tuition_fees_usd', 'living_cost_usd', 'min_gpa', 
            'acceptance_rate', 'reputation_score', 'international_student_ratio'
        ]
        for col in numeric_columns:
            if col in self.universities_df.columns:
                self.universities_df[col] = pd.to_numeric(self.universities_df[col], errors='coerce')
    
    def _parse_test_scores(self, score_str: str) -> Dict[str, int]:
        """Parse test score string into dictionary.
        
        Args:
            score_str: String like "SAT:1200;TOEFL:90"
            
        Returns:
            Dictionary of test scores
        """
        if pd.isna(score_str) or not isinstance(score_str, str):
            return {}
        
        try:
            scores = {}
            for score_pair in score_str.split(';'):
                if ':' in score_pair:
                    test, score = score_pair.split(':')
                    scores[test.strip()] = int(score.strip())
            return scores
        except:
            return {}
    
    def _convert_to_models(self) -> None:
        """Convert DataFrame rows to University model objects."""
        self.universities = []
        
        if self.universities_df is None:
            return
            
        for _, row in self.universities_df.iterrows():
            try:
                university = University(
                    university_id=str(row.get('university_id', f"univ_{len(self.universities)}")),
                    name=str(row['name']),
                    country=str(row['country']),
                    city=str(row['city']),
                    ranking_global=int(row.get('ranking_global') or 9999) if row.get('ranking_global') is not None and str(row.get('ranking_global')) != 'nan' else None,
                    ranking_national=int(row.get('ranking_national') or 9999) if row.get('ranking_national') is not None and str(row.get('ranking_national')) != 'nan' else None,
                    reputation_score=float(row.get('reputation_score') or 50.0),
                    tuition_fees_usd=float(row['tuition_fees_usd'] or 0),
                    living_cost_usd=float(row['living_cost_usd'] or 0),
                    min_gpa=float(row['min_gpa'] or 0),
                    min_test_scores=row.get('min_test_scores') or {},
                    acceptance_rate=float(row['acceptance_rate'] or 0.5),
                    international_student_ratio=float(row.get('international_student_ratio') or 0.1),
                    student_population=int(row.get('student_population') or 10000),
                    programs_offered=row.get('programs_offered') or [],
                    languages_taught=row.get('languages_taught') or ['English'],
                    university_type=UniversityType(row.get('university_type') or 'public'),
                    campus_culture_tags=[CampusCulture(tag) for tag in (row.get('campus_culture_tags') or []) 
                                       if tag in [c.value for c in CampusCulture]],
                    has_scholarship=bool(row.get('has_scholarship', False)),
                    research_opportunities=bool(row.get('research_opportunities', True))
                )
                self.universities.append(university)
            except Exception as e:
                logger.warning(f"Error converting row to University model: {e}")
                continue
    
    def _create_sample_data(self) -> None:
        """Create sample university data for testing."""
        sample_universities = [
            {
                'university_id': 'mit_001',
                'name': 'Massachusetts Institute of Technology',
                'country': 'USA',
                'city': 'Cambridge',
                'ranking_global': 1,
                'ranking_national': 1,
                'reputation_score': 98.5,
                'tuition_fees_usd': 53790,
                'living_cost_usd': 18000,
                'min_gpa': 3.8,
                'min_test_scores': {'SAT': 1520, 'TOEFL': 100},
                'acceptance_rate': 0.07,
                'international_student_ratio': 0.35,
                'student_population': 11254,
                'programs_offered': ['Computer Science', 'Engineering', 'Physics', 'Mathematics'],
                'languages_taught': ['English'],
                'university_type': 'private',
                'campus_culture_tags': ['research_focused', 'technical'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'stanford_001',
                'name': 'Stanford University',
                'country': 'USA',
                'city': 'Stanford',
                'ranking_global': 2,
                'ranking_national': 2,
                'reputation_score': 97.8,
                'tuition_fees_usd': 56169,
                'living_cost_usd': 19500,
                'min_gpa': 3.7,
                'min_test_scores': {'SAT': 1500, 'TOEFL': 100},
                'acceptance_rate': 0.04,
                'international_student_ratio': 0.30,
                'student_population': 17249,
                'programs_offered': ['Computer Science', 'Business', 'Engineering', 'Medicine'],
                'languages_taught': ['English'],
                'university_type': 'private',
                'campus_culture_tags': ['research_focused', 'diverse'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'oxford_001',
                'name': 'University of Oxford',
                'country': 'UK',
                'city': 'Oxford',
                'ranking_global': 3,
                'ranking_national': 1,
                'reputation_score': 97.2,
                'tuition_fees_usd': 45000,
                'living_cost_usd': 15000,
                'min_gpa': 3.8,
                'min_test_scores': {'IELTS': 7, 'TOEFL': 100},
                'acceptance_rate': 0.15,
                'international_student_ratio': 0.40,
                'student_population': 24000,
                'programs_offered': ['Liberal Arts', 'Sciences', 'Medicine', 'Law'],
                'languages_taught': ['English'],
                'university_type': 'public',
                'campus_culture_tags': ['traditional', 'research_focused'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'utoronto_001',
                'name': 'University of Toronto',
                'country': 'Canada',
                'city': 'Toronto',
                'ranking_global': 25,
                'ranking_national': 1,
                'reputation_score': 85.4,
                'tuition_fees_usd': 35000,
                'living_cost_usd': 12000,
                'min_gpa': 3.5,
                'min_test_scores': {'TOEFL': 90, 'IELTS': 6.5},
                'acceptance_rate': 0.43,
                'international_student_ratio': 0.25,
                'student_population': 97000,
                'programs_offered': ['Computer Science', 'Engineering', 'Business', 'Arts'],
                'languages_taught': ['English'],
                'university_type': 'public',
                'campus_culture_tags': ['diverse', 'research_focused'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'epfl_001',
                'name': 'École Polytechnique Fédérale de Lausanne',
                'country': 'Switzerland',
                'city': 'Lausanne',
                'ranking_global': 14,
                'ranking_national': 1,
                'reputation_score': 90.2,
                'tuition_fees_usd': 1500,
                'living_cost_usd': 20000,
                'min_gpa': 3.6,
                'min_test_scores': {'TOEFL': 95, 'IELTS': 7},
                'acceptance_rate': 0.25,
                'international_student_ratio': 0.60,
                'student_population': 12000,
                'programs_offered': ['Engineering', 'Computer Science', 'Physics', 'Mathematics'],
                'languages_taught': ['English', 'French'],
                'university_type': 'public',
                'campus_culture_tags': ['technical', 'research_focused'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'tu_delft_001',
                'name': 'Delft University of Technology',
                'country': 'Netherlands',
                'city': 'Delft',
                'ranking_global': 50,
                'ranking_national': 1,
                'reputation_score': 82.5,
                'tuition_fees_usd': 18000,
                'living_cost_usd': 14000,
                'min_gpa': 3.3,
                'min_test_scores': {'TOEFL': 90, 'IELTS': 6.5},
                'acceptance_rate': 0.35,
                'international_student_ratio': 0.35,
                'student_population': 26000,
                'programs_offered': ['Engineering', 'Computer Science', 'Architecture'],
                'languages_taught': ['English', 'Dutch'],
                'university_type': 'public',
                'campus_culture_tags': ['technical', 'research_focused'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'melbourne_001',
                'name': 'University of Melbourne',
                'country': 'Australia',
                'city': 'Melbourne',
                'ranking_global': 35,
                'ranking_national': 1,
                'reputation_score': 87.2,
                'tuition_fees_usd': 42000,
                'living_cost_usd': 16000,
                'min_gpa': 3.4,
                'min_test_scores': {'IELTS': 6.5, 'TOEFL': 79},
                'acceptance_rate': 0.28,
                'international_student_ratio': 0.40,
                'student_population': 50000,
                'programs_offered': ['Medicine', 'Engineering', 'Business', 'Arts'],
                'languages_taught': ['English'],
                'university_type': 'public',
                'campus_culture_tags': ['diverse', 'research_focused'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'tsinghua_001',
                'name': 'Tsinghua University',
                'country': 'China',
                'city': 'Beijing',
                'ranking_global': 18,
                'ranking_national': 1,
                'reputation_score': 92.1,
                'tuition_fees_usd': 4500,
                'living_cost_usd': 8000,
                'min_gpa': 3.7,
                'min_test_scores': {'TOEFL': 100, 'HSK': 6},
                'acceptance_rate': 0.12,
                'international_student_ratio': 0.15,
                'student_population': 47000,
                'programs_offered': ['Engineering', 'Computer Science', 'Business', 'Architecture'],
                'languages_taught': ['English', 'Chinese'],
                'university_type': 'public',
                'campus_culture_tags': ['technical', 'research_focused'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'mcgill_001',
                'name': 'McGill University',
                'country': 'Canada',
                'city': 'Montreal',
                'ranking_global': 32,
                'ranking_national': 2,
                'reputation_score': 84.8,
                'tuition_fees_usd': 25000,
                'living_cost_usd': 11000,
                'min_gpa': 3.3,
                'min_test_scores': {'TOEFL': 86, 'IELTS': 6.5},
                'acceptance_rate': 0.46,
                'international_student_ratio': 0.30,
                'student_population': 40000,
                'programs_offered': ['Medicine', 'Engineering', 'Arts', 'Science'],
                'languages_taught': ['English', 'French'],
                'university_type': 'public',
                'campus_culture_tags': ['diverse', 'traditional'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'tub_001',
                'name': 'Technical University of Berlin',
                'country': 'Germany',
                'city': 'Berlin',
                'ranking_global': 85,
                'ranking_national': 5,
                'reputation_score': 78.2,
                'tuition_fees_usd': 500,  # Very low tuition
                'living_cost_usd': 12000,
                'min_gpa': 3.0,
                'min_test_scores': {'TOEFL': 80, 'IELTS': 6.0},
                'acceptance_rate': 0.45,
                'international_student_ratio': 0.25,
                'student_population': 35000,
                'programs_offered': ['Engineering', 'Computer Science', 'Mathematics'],
                'languages_taught': ['English', 'German'],
                'university_type': 'public',
                'campus_culture_tags': ['technical', 'diverse'],
                'has_scholarship': True,
                'research_opportunities': True
            },
            {
                'university_id': 'ubc_001',
                'name': 'University of British Columbia',
                'country': 'Canada',
                'city': 'Vancouver',
                'ranking_global': 40,
                'ranking_national': 3,
                'reputation_score': 83.5,
                'tuition_fees_usd': 28000,
                'living_cost_usd': 13000,
                'min_gpa': 3.2,
                'min_test_scores': {'TOEFL': 90, 'IELTS': 6.5},
                'acceptance_rate': 0.52,
                'international_student_ratio': 0.35,
                'student_population': 45000,
                'programs_offered': ['Engineering', 'Computer Science', 'Business', 'Medicine'],
                'languages_taught': ['English'],
                'university_type': 'public',
                'campus_culture_tags': ['diverse', 'research_focused'],
                'has_scholarship': True,
                'research_opportunities': True
            }
        ]
        
        self.universities_df = pd.DataFrame(sample_universities)
        logger.info(f"Created sample dataset with {len(sample_universities)} universities")
        self._preprocess_data()
        self._convert_to_models()
        
        # Save sample data to file
        Path("data").mkdir(exist_ok=True)
        self.universities_df.to_csv(self.data_file_path, index=False)
        logger.info(f"Saved sample data to {self.data_file_path}")
    
    def get_universities(self) -> List[University]:
        """Get all universities as model objects.
        
        Returns:
            List of University model objects
        """
        return self.universities
    
    def get_university_by_id(self, university_id: str) -> Optional[University]:
        """Get a specific university by ID.
        
        Args:
            university_id: University identifier
            
        Returns:
            University object or None if not found
        """
        for university in self.universities:
            if university.university_id == university_id:
                return university
        return None
    
    def filter_universities(self, 
                          countries: Optional[List[str]] = None,
                          max_tuition: Optional[float] = None,
                          min_gpa_requirement: Optional[float] = None,
                          programs: Optional[List[str]] = None) -> List[University]:
        """Filter universities based on criteria.
        
        Args:
            countries: List of acceptable countries
            max_tuition: Maximum tuition fees
            min_gpa_requirement: Minimum GPA requirement filter
            programs: Required programs
            
        Returns:
            Filtered list of universities
        """
        filtered = self.universities.copy()
        
        if countries:
            filtered = [u for u in filtered if u.country in countries]
        
        if max_tuition is not None:
            filtered = [u for u in filtered if u.tuition_fees_usd <= max_tuition]
        
        if min_gpa_requirement is not None:
            filtered = [u for u in filtered if u.min_gpa <= min_gpa_requirement]
        
        if programs:
            filtered = [u for u in filtered 
                       if any(prog.lower() in [p.lower() for p in u.programs_offered] 
                             for prog in programs)]
        
        return filtered
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.universities:
            return {}
        
        tuitions = [u.tuition_fees_usd for u in self.universities]
        gpas = [u.min_gpa for u in self.universities]
        
        return {
            'total_universities': len(self.universities),
            'countries': list(set(u.country for u in self.universities)),
            'avg_tuition': np.mean(tuitions),
            'median_tuition': np.median(tuitions),
            'avg_min_gpa': np.mean(gpas),
            'programs_available': list(set(
                prog for u in self.universities for prog in u.programs_offered
            ))
        } 