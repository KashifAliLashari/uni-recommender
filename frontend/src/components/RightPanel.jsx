import RecommendationCard from './RecommendationCard';
import SkeletonCard from './SkeletonCard';

const RightPanel = ({ recommendations, loading, error, hasSubmitted }) => {
    return (
        <div className="bg-white p-6 rounded-lg shadow-xl border border-gray-200 h-full">
            <h2 className="text-xl font-semibold mb-4 text-black">University Recommendations</h2>

            {loading && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {[...Array(4)].map((_, i) => <SkeletonCard key={i} />)}
                </div>
            )}

            {error && (
                <div className="text-red-500 bg-red-100 p-4 rounded-md border border-red-300">
                    <p className="font-semibold">Error</p>
                    <p>{error}</p>
                </div>
            )}

            {!loading && !error && hasSubmitted && recommendations && recommendations.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {recommendations.map((recommendation, index) => (
                        <RecommendationCard key={index} recommendation={recommendation} />
                    ))}
                </div>
            )}

            {!loading && !error && hasSubmitted && (!recommendations || recommendations.length === 0) && (
                <div className="text-center text-gray-500 py-16">
                    <h3 className="text-lg font-semibold">No recommendations found</h3>
                    <p>Try adjusting your preferences and search again.</p>
                </div>
            )}

            {!hasSubmitted && (
                <div className="text-center text-gray-500 py-16">
                    <div className="max-w-md mx-auto">
                        <div className="mb-4">
                            <svg className="w-16 h-16 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                            </svg>
                        </div>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">Ready to find your perfect university?</h3>
                        <p className="text-sm">Fill out your profile on the left and click "Get University Recommendations" to see personalized matches based on your preferences and qualifications.</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default RightPanel; 