"""
Machine learning models for university recommendations.
Implements collaborative filtering and content-based filtering.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import logging

from .models import StudentProfile, University, MatchScore

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based filtering for university recommendations."""
    
    def __init__(self):
        """Initialize the content-based recommender."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            min_df=1,  # Allow words that appear in only 1 document
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.university_features = None
        self.university_similarity_matrix = None
        self.is_trained = False
    
    def train(self, universities: List[University]) -> None:
        """Train the content-based model on university data.
        
        Args:
            universities: List of universities for training
        """
        if not universities:
            logger.warning("No universities provided for training")
            return
        
        try:
            # Create feature matrix
            features_df = self._create_feature_matrix(universities)
            
            # Calculate similarity matrix
            self.university_similarity_matrix = cosine_similarity(features_df)
            self.university_features = features_df
            self.is_trained = True
            
            logger.info(f"Content-based model trained on {len(universities)} universities")
            
        except Exception as e:
            logger.error(f"Error training content-based model: {e}")
            self.is_trained = False
    
    def _create_feature_matrix(self, universities: List[University]) -> pd.DataFrame:
        """Create feature matrix from university data.
        
        Args:
            universities: List of universities
            
        Returns:
            Feature matrix as DataFrame
        """
        features = []
        
        for university in universities:
            # Text features (programs offered)
            programs_text = ' '.join(university.programs_offered)
            
            # Categorical features
            country = university.country
            university_type = university.university_type.value
            
            # Numerical features
            features.append({
                'programs_text': programs_text,
                'country': country,
                'university_type': university_type,
                'tuition_fees_usd': university.tuition_fees_usd,
                'living_cost_usd': university.living_cost_usd,
                'min_gpa': university.min_gpa,
                'acceptance_rate': university.acceptance_rate,
                'reputation_score': university.reputation_score,
                'international_student_ratio': university.international_student_ratio,
                'student_population': university.student_population,
                'research_opportunities': int(university.research_opportunities),
                'has_scholarship': int(university.has_scholarship)
            })
        
        df = pd.DataFrame(features)
        
        # Process text features
        if 'programs_text' in df.columns and not df['programs_text'].empty:
            # Filter out empty or whitespace-only texts
            non_empty_texts = df['programs_text'].fillna('').str.strip()
            non_empty_texts = non_empty_texts[non_empty_texts != '']
            
            if len(non_empty_texts) > 0:
                try:
                    from scipy.sparse import csr_matrix
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['programs_text'].fillna('unknown'))
                    try:
                        tfidf_array = tfidf_matrix.toarray()  # type: ignore
                    except AttributeError:
                        tfidf_array = tfidf_matrix
                    tfidf_df = pd.DataFrame(
                        tfidf_array, 
                        columns=[f'program_tfidf_{i}' for i in range(tfidf_array.shape[1])]  # type: ignore
                    )
                except ValueError as e:
                    # If TF-IDF fails, create dummy features
                    logger.warning(f"TF-IDF failed, using dummy features: {e}")
                    tfidf_df = pd.DataFrame({'program_dummy': [1] * len(df)})
            else:
                tfidf_df = pd.DataFrame({'program_dummy': [1] * len(df)})
        else:
            tfidf_df = pd.DataFrame({'program_dummy': [1] * len(df)})
        
        # Process categorical features
        categorical_features = []
        for col in ['country', 'university_type']:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df[col])
                categorical_features.append(pd.DataFrame({f'{col}_encoded': encoded}))
        
        # Process numerical features
        numerical_cols = [
            'tuition_fees_usd', 'living_cost_usd', 'min_gpa', 'acceptance_rate',
            'reputation_score', 'international_student_ratio', 'student_population',
            'research_opportunities', 'has_scholarship'
        ]
        numerical_df = df[numerical_cols].fillna(0)
        numerical_scaled = self.scaler.fit_transform(numerical_df)
        numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_cols)  # type: ignore
        
        # Combine all features
        combined_features = [numerical_scaled_df]
        if not tfidf_df.empty:
            combined_features.append(tfidf_df)
        combined_features.extend(categorical_features)
        
        return pd.concat(combined_features, axis=1)
    
    def get_similar_universities(self, 
                               target_university_idx: int, 
                               universities: List[University],
                               top_k: int = 10) -> List[Tuple[University, float]]:
        """Get universities similar to a target university.
        
        Args:
            target_university_idx: Index of target university
            universities: List of all universities
            top_k: Number of similar universities to return
            
        Returns:
            List of (University, similarity_score) tuples
        """
        if not self.is_trained or self.university_similarity_matrix is None:
            logger.warning("Model not trained")
            return []
        
        if target_university_idx >= len(universities):
            logger.warning("Invalid university index")
            return []
        
        # Get similarity scores for target university
        similarity_scores = self.university_similarity_matrix[target_university_idx]
        
        # Get top-k similar universities (excluding the target itself)
        similar_indices = np.argsort(similarity_scores)[::-1]
        similar_universities = []
        
        for idx in similar_indices:
            if idx != target_university_idx and len(similar_universities) < top_k:
                similarity_score = similarity_scores[idx]
                similar_universities.append((universities[idx], similarity_score))
        
        return similar_universities
    
    def recommend_based_on_preferences(self, 
                                     student: StudentProfile,
                                     universities: List[University],
                                     top_k: int = 20) -> List[Tuple[University, float]]:
        """Recommend universities based on student preferences using content similarity.
        
        Args:
            student: Student profile
            universities: List of universities
            top_k: Number of recommendations
            
        Returns:
            List of (University, content_score) tuples
        """
        if not self.is_trained:
            logger.warning("Model not trained")
            return [(u, 0.5) for u in universities[:top_k]]
        
        # Create a pseudo-university from student preferences
        student_features = self._create_student_feature_vector(student)
        
        if student_features is None:
            return [(u, 0.5) for u in universities[:top_k]]
        
        # Calculate similarity between student preferences and all universities
        similarity_scores = cosine_similarity([student_features], self.university_features)[0]
        
        # Sort universities by similarity score
        university_scores = list(zip(universities, similarity_scores))
        university_scores.sort(key=lambda x: x[1], reverse=True)
        
        return university_scores[:top_k]
    
    def _create_student_feature_vector(self, student: StudentProfile) -> Optional[np.ndarray]:
        """Create a feature vector representing student preferences.
        
        Args:
            student: Student profile
            
        Returns:
            Feature vector or None if creation fails
        """
        try:
            # Create a pseudo-feature vector based on student preferences
            features = {
                'programs_text': student.field_of_study,
                'country': student.preferred_countries[0] if student.preferred_countries else 'USA',
                'university_type': student.university_type_preference.value if student.university_type_preference else 'public',
                'tuition_fees_usd': student.budget_max * 0.7,  # Assume 70% of budget for tuition
                'living_cost_usd': student.budget_max * 0.3,   # 30% for living costs
                'min_gpa': student.gpa - 0.2,  # Prefer slightly lower requirements
                'acceptance_rate': 0.3,  # Prefer reasonable acceptance rates
                'reputation_score': 80.0,  # Prefer good reputation
                'international_student_ratio': 0.2,
                'student_population': 15000,  # Medium-sized universities
                'research_opportunities': 1,
                'has_scholarship': 1
            }
            
            df = pd.DataFrame([features])
            
            # Process similar to training data
            if hasattr(self.tfidf_vectorizer, 'vocabulary_') and len(self.tfidf_vectorizer.vocabulary_) > 0:
                try:
                    tfidf_matrix = self.tfidf_vectorizer.transform(df['programs_text'])
                    try:
                        tfidf_array = tfidf_matrix.toarray()  # type: ignore
                    except AttributeError:
                        tfidf_array = tfidf_matrix
                    tfidf_features = tfidf_array[0]
                except:
                    # If transform fails, use dummy features
                    tfidf_features = np.array([1.0])  # Single dummy feature
            else:
                tfidf_features = np.array([1.0])  # Single dummy feature
            
            # Process categorical features
            categorical_features = []
            for col in ['country', 'university_type']:
                if col in self.label_encoders:
                    try:
                        encoded = self.label_encoders[col].transform(df[col])[0]
                        categorical_features.append(encoded)
                    except ValueError:
                        # Handle unseen categories
                        categorical_features.append(0)
                else:
                    categorical_features.append(0)
            
            # Process numerical features
            numerical_cols = [
                'tuition_fees_usd', 'living_cost_usd', 'min_gpa', 'acceptance_rate',
                'reputation_score', 'international_student_ratio', 'student_population',
                'research_opportunities', 'has_scholarship'
            ]
            numerical_data = df[numerical_cols].fillna(0).values[0]
            numerical_scaled = self.scaler.transform([numerical_data])[0]
            
            # Combine all features
            feature_vector = np.concatenate([
                numerical_scaled,
                tfidf_features,
                categorical_features
            ])
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error creating student feature vector: {e}")
            return None


class CollaborativeFilteringRecommender:
    """Collaborative filtering for university recommendations."""
    
    def __init__(self, n_components: int = 50):
        """Initialize the collaborative filtering recommender.
        
        Args:
            n_components: Number of components for SVD
        """
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.student_university_matrix = None
        self.student_features = None
        self.university_features = None
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.is_trained = False
    
    def train(self, universities: List[University], synthetic_data: bool = True) -> None:
        """Train the collaborative filtering model.
        
        Args:
            universities: List of universities
            synthetic_data: Whether to generate synthetic interaction data
        """
        try:
            if synthetic_data:
                # Generate synthetic student-university interaction data
                self._generate_synthetic_data(universities)
            
            if self.student_university_matrix is not None and self.student_university_matrix.shape[1] > 0:
                # Ensure we don't have more components than features
                n_components = min(self.n_components, self.student_university_matrix.shape[1] - 1, 
                                 self.student_university_matrix.shape[0] - 1)
                if n_components > 0:
                    self.svd.n_components = n_components
                    
                    # Apply SVD for dimensionality reduction
                    self.student_features = self.svd.fit_transform(self.student_university_matrix)
                    self.university_features = self.svd.components_.T
                    
                    # Train KNN model for finding similar students
                    self.knn_model.fit(self.student_features)
                    
                    self.is_trained = True
                    logger.info(f"Collaborative filtering model trained with {self.student_university_matrix.shape[0]} students and {self.student_university_matrix.shape[1]} universities")
                else:
                    logger.warning("Insufficient data for collaborative filtering training")
            else:
                logger.warning("No student-university interaction matrix available")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
            self.is_trained = False
    
    def _generate_synthetic_data(self, universities: List[University]) -> None:
        """Generate synthetic student-university interaction data.
        
        Args:
            universities: List of universities
        """
        n_students = 1000
        n_universities = len(universities)
        
        # Create interaction matrix (students x universities)
        # Use ratings from 1-5, where 0 means no interaction
        interaction_matrix = np.zeros((n_students, n_universities))
        
        for student_idx in range(n_students):
            # Each student interacts with 3-15 universities (but not more than available)
            max_interactions = min(15, n_universities)
            min_interactions = min(3, n_universities)
            n_interactions = np.random.randint(min_interactions, max_interactions + 1)
            university_indices = np.random.choice(n_universities, n_interactions, replace=False)
            
            for univ_idx in university_indices:
                university = universities[univ_idx]
                
                # Generate realistic ratings based on university characteristics
                base_rating = 3.0
                
                # Adjust based on reputation
                if university.reputation_score > 90:
                    base_rating += 1.0
                elif university.reputation_score > 70:
                    base_rating += 0.5
                
                # Adjust based on acceptance rate (more selective = higher rating)
                if university.acceptance_rate < 0.1:
                    base_rating += 0.5
                elif university.acceptance_rate > 0.7:
                    base_rating -= 0.3
                
                # Add some randomness
                rating = base_rating + np.random.normal(0, 0.5)
                rating = np.clip(rating, 1, 5)
                
                interaction_matrix[student_idx, univ_idx] = rating
        
        self.student_university_matrix = interaction_matrix
    
    def find_similar_students(self, student_profile: StudentProfile, 
                            top_k: int = 5) -> List[int]:
        """Find students similar to the given student profile.
        
        Args:
            student_profile: Target student profile
            top_k: Number of similar students to find
            
        Returns:
            List of similar student indices
        """
        if not self.is_trained:
            return []
        
        # Create a feature vector for the target student
        student_vector = self._create_student_vector(student_profile)
        
        # Find similar students
        distances, indices = self.knn_model.kneighbors([student_vector], n_neighbors=top_k)
        
        return indices[0].tolist()
    
    def _create_student_vector(self, student: StudentProfile) -> np.ndarray:
        """Create a feature vector for a student.
        
        Args:
            student: Student profile
            
        Returns:
            Student feature vector
        """
        if not self.is_trained or self.student_features is None:
            # Return a simple vector if not trained
            return np.array([0.5] * 5)
        
        # Create a feature vector that matches the dimensionality of trained features
        n_features = self.student_features.shape[1]
        
        # Create a simplified feature vector based on student characteristics
        features = [
            student.gpa / 4.0,  # Normalized GPA
            min(student.budget_max / 100000, 1.0),  # Normalized budget
            len(student.preferred_countries) / 10,  # Number of preferred countries
            1.0 if student.university_type_preference else 0.0,  # Has type preference
            len(student.test_scores) / 5,  # Number of test scores
        ]
        
        # Pad or truncate to match the exact number of features from training
        while len(features) < n_features:
            features.append(0.0)
        
        return np.array(features[:n_features])
    
    def recommend_universities(self, student_profile: StudentProfile,
                             universities: List[University],
                             top_k: int = 20) -> List[Tuple[University, float]]:
        """Recommend universities using collaborative filtering.
        
        Args:
            student_profile: Student profile
            universities: List of universities
            top_k: Number of recommendations
            
        Returns:
            List of (University, collaborative_score) tuples
        """
        if not self.is_trained:
            # Return random recommendations if not trained
            indices = np.random.choice(len(universities), min(top_k, len(universities)), replace=False)
            return [(universities[i], 0.5) for i in indices]
        
        # Find similar students
        similar_students = self.find_similar_students(student_profile, top_k=10)
        
        if not similar_students:
            # Return random recommendations if no similar students found
            indices = np.random.choice(len(universities), min(top_k, len(universities)), replace=False)
            return [(universities[i], 0.5) for i in indices]
        
        # Calculate recommendation scores based on similar students' preferences
        university_scores = np.zeros(len(universities))
        
        for student_idx in similar_students:
            student_ratings = self.student_university_matrix[student_idx]  # type: ignore
            university_scores += student_ratings
        
        # Normalize scores
        university_scores = university_scores / len(similar_students)
        
        # Create university-score pairs and sort
        university_score_pairs = [(universities[i], university_scores[i]) 
                                for i in range(len(universities))]
        university_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return university_score_pairs[:top_k]


class HybridRecommender:
    """Hybrid recommender combining content-based and collaborative filtering."""
    
    def __init__(self, content_weight: float = 0.6, collaborative_weight: float = 0.4):
        """Initialize the hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations
            collaborative_weight: Weight for collaborative filtering recommendations
        """
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.is_trained = False
    
    def train(self, universities: List[University]) -> None:
        """Train both recommendation models.
        
        Args:
            universities: List of universities for training
        """
        try:
            # Train content-based model
            self.content_recommender.train(universities)
            
            # Train collaborative filtering model
            self.collaborative_recommender.train(universities, synthetic_data=True)
            
            self.is_trained = (self.content_recommender.is_trained and 
                             self.collaborative_recommender.is_trained)
            
            if self.is_trained:
                logger.info("Hybrid recommender trained successfully")
            else:
                logger.warning("Hybrid recommender training incomplete")
                
        except Exception as e:
            logger.error(f"Error training hybrid recommender: {e}")
            self.is_trained = False
    
    def recommend(self, student_profile: StudentProfile,
                  universities: List[University],
                  top_k: int = 20) -> List[Tuple[University, float]]:
        """Generate hybrid recommendations.
        
        Args:
            student_profile: Student profile
            universities: List of universities
            top_k: Number of recommendations
            
        Returns:
            List of (University, hybrid_score) tuples
        """
        if not universities:
            return []
        
        # Get content-based recommendations
        content_recommendations = self.content_recommender.recommend_based_on_preferences(
            student_profile, universities, top_k=len(universities)
        )
        
        # Get collaborative filtering recommendations
        collaborative_recommendations = self.collaborative_recommender.recommend_universities(
            student_profile, universities, top_k=len(universities)
        )
        
        # Create score dictionaries for easier lookup
        content_scores = {univ.university_id: score for univ, score in content_recommendations}
        collaborative_scores = {univ.university_id: score for univ, score in collaborative_recommendations}
        
        # Calculate hybrid scores
        hybrid_recommendations = []
        for university in universities:
            content_score = content_scores.get(university.university_id, 0.0)
            collaborative_score = collaborative_scores.get(university.university_id, 0.0)
            
            # Combine scores with weights
            hybrid_score = (self.content_weight * content_score + 
                          self.collaborative_weight * collaborative_score)
            
            hybrid_recommendations.append((university, hybrid_score))
        
        # Sort by hybrid score and return top-k
        hybrid_recommendations.sort(key=lambda x: x[1], reverse=True)
        return hybrid_recommendations[:top_k] 