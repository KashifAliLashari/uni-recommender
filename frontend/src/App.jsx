import { useState } from 'react';
import Header from './components/Header';
import LeftPanel from './components/LeftPanel';
import RightPanel from './components/RightPanel';
import { getRecommendations } from './services/api';

function App() {
  const [formState, setFormState] = useState({
    applicationType: 'undergraduate',
    gpa: 3.0,
    budget: 50000,
    priorities: {
      reputation: 25,
      cost: 25,
      location: 25,
      culture: 25,
    },
    // Add other form fields as well
    testType: 'SAT',
    testScore: 1200,
    educationLevel: 'High School',
    achievements: '',
    countries: [],
    fieldOfStudy: '',
    timeline: 'Fall 2025'
  });

  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setHasSubmitted(true);

    try {
      const data = await getRecommendations(formState);
      setRecommendations(data.recommendations);
    } catch (err) {
      setError('Failed to fetch recommendations. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen text-black font-sans" style={{ background: 'linear-gradient(135deg, #5efeff 0%, #ff7575 100%)' }}>
      <Header />
      <main className="flex flex-col md:flex-row flex-1 p-4 md:p-6 lg:p-8 gap-6">
        <div className="w-full md:w-2/5">
          <LeftPanel
            formState={formState}
            setFormState={setFormState}
            onSubmit={handleSubmit}
            loading={loading}
          />
        </div>
        <div className="w-full md:w-3/5">
          <RightPanel
            recommendations={recommendations}
            loading={loading}
            error={error}
            hasSubmitted={hasSubmitted}
          />
        </div>
      </main>
      {/* Bottom section for comparison table will be added later */}
    </div>
  );
}

export default App;
