import { useState } from 'react';
import { Heart, ChevronDown, CheckSquare } from 'lucide-react';
import CircularProgress from './CircularProgress';
import { motion, AnimatePresence } from 'framer-motion';

const RecommendationCard = ({ recommendation }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    // Extract data from the recommendation object
    const university = recommendation.university;
    const matchScore = recommendation.match_score;
    const explanation = recommendation.explanation;

    // Fallback for missing data
    const safeData = {
        ...university,
        ...matchScore,
        name: university?.name || 'Unknown University',
        country: university?.country || 'Unknown Location',
        city: university?.city || '',
        ranking_global: university?.ranking_global || 'N/A',
        tuition_fees_usd: university?.tuition_fees_usd || 0,
        acceptance_rate: (university?.acceptance_rate * 100) || 'N/A',
        overall_score: matchScore?.overall_score || 0
    };

    const location = safeData.city ? `${safeData.city}, ${safeData.country}` : safeData.country;

    return (
        <div className="bg-white rounded-lg shadow-xl border border-gray-200 p-4 transition-all duration-300 hover:shadow-2xl hover:bg-gray-50">
            <div className="flex items-start gap-4">
                <div className="w-24 h-24 flex-shrink-0">
                    <CircularProgress percentage={safeData.overall_score} />
                </div>
                <div className="flex-grow">
                    <h3 className="text-xl font-bold text-black">{safeData.name}</h3>
                    <p className="text-sm text-gray-500">{location}</p>
                    <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                        <span>Rank: <span className="text-black font-semibold">#{safeData.ranking_global}</span></span>
                        <span>Tuition: <span className="text-black font-semibold">${safeData.tuition_fees_usd?.toLocaleString()}</span></span>
                        <span>Acceptance: <span className="text-black font-semibold">{Math.round(safeData.acceptance_rate)}%</span></span>
                    </div>
                </div>
            </div>

            <div className="mt-4">
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="text-sm text-blue-400 hover:underline flex items-center"
                >
                    Why this match?
                    <motion.div animate={{ rotate: isExpanded ? 180 : 0 }} transition={{ duration: 0.3 }}>
                        <ChevronDown className="h-4 w-4 ml-1" />
                    </motion.div>
                </button>
                <AnimatePresence>
                    {isExpanded && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                        >
                            <div className="mt-2 text-sm text-gray-800 space-y-2 border-t border-gray-200 pt-2">
                                <p className="mb-2">{explanation}</p>
                                <div className="grid grid-cols-2 gap-2 text-xs">
                                    <p><strong className="text-gray-500">Academic Fit:</strong> {Math.round(safeData.academic_fit || 0)}%</p>
                                    <p><strong className="text-gray-500">Financial Fit:</strong> {Math.round(safeData.financial_feasibility || 0)}%</p>
                                    <p><strong className="text-gray-500">Preference Match:</strong> {Math.round(safeData.preference_alignment || 0)}%</p>
                                    <p><strong className="text-gray-500">Admission Chance:</strong> {Math.round(safeData.admission_probability || 0)}%</p>
                                </div>
                                {recommendation.total_annual_cost && (
                                    <p className="mt-2"><strong className="text-gray-500">Total Annual Cost:</strong> ${Math.round(recommendation.total_annual_cost).toLocaleString()}</p>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

export default RecommendationCard; 