import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

// Mapping from frontend country codes to backend country names
const COUNTRY_CODE_TO_NAME = {
    'USA': 'USA',
    'CAN': 'Canada',
    'GBR': 'UK',
    'AUS': 'Australia',
    'DEU': 'Germany',
    'NLD': 'Netherlands',
    'CHE': 'Switzerland',
    'CHN': 'China'
};

export const getRecommendations = async (formData) => {
    try {
        // Transform frontend form data to backend API format
        const requestPayload = {
            student_profile: {
                gpa: formData.applicationType === 'graduate' ? formData.gpa : 3.5,
                test_scores: {
                    [formData.testType]: formData.testType === 'IELTS' ?
                        parseFloat(formData.testScore) || 6.5 :
                        parseInt(formData.testScore) ||
                        (formData.testType === 'SAT' ? 1200 :
                            formData.testType === 'ACT' ? 24 :
                                formData.testType === 'TOEFL' ? 79 :
                                    formData.testType === 'GRE' ? 310 : 0)
                },
                qualifications: formData.achievements ? [formData.achievements] : [],
                preferred_countries: formData.countries?.map(c => COUNTRY_CODE_TO_NAME[c.value] || c.value) || [],
                preferred_cities: [],
                budget_max: formData.budget,
                field_of_study: formData.fieldOfStudy || "",
                language_requirements: {},
                timeline: formData.timeline,
                priorities: {
                    academic_reputation: formData.priorities.reputation / 100 * 0.9,
                    cost: formData.priorities.cost / 100 * 0.9,
                    location: formData.priorities.location / 100 * 0.9,
                    culture: formData.priorities.culture / 100 * 0.9,
                    admission_probability: 0.1
                },
                university_size_preference: null,
                university_type_preference: null,
                campus_culture_preferences: []
            },
            max_recommendations: 10,
            include_explanations: true
        };

        console.log('Sending request payload:', JSON.stringify(requestPayload, null, 2));

        const response = await axios.post(`${API_URL}/recommendations`, requestPayload);
        return response.data;
    } catch (error) {
        console.error("Error fetching recommendations:", error);

        // Log detailed error information for debugging
        if (error.response) {
            console.error("Response status:", error.response.status);
            console.error("Response data:", error.response.data);
            console.error("Response headers:", error.response.headers);
        }

        throw error;
    }
}; 