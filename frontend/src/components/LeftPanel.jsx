import StaticSection from './StaticSection';
import Select from 'react-select';

const countryOptions = [
    { value: 'USA', label: '🇺🇸 United States' },
    { value: 'CAN', label: '🇨🇦 Canada' },
    { value: 'GBR', label: '🇬🇧 United Kingdom' },
    { value: 'AUS', label: '🇦🇺 Australia' },
    { value: 'DEU', label: '🇩🇪 Germany' },
    { value: 'NLD', label: '🇳🇱 Netherlands' },
    { value: 'CHE', label: '🇨🇭 Switzerland' },
    { value: 'CHN', label: '🇨🇳 China' },
];

const customStyles = {
    control: (provided) => ({
        ...provided,
        backgroundColor: '#374151',
        borderColor: '#4B5563',
        color: 'white',
    }),
    menu: (provided) => ({
        ...provided,
        backgroundColor: '#1F2937',
    }),
    option: (provided, state) => ({
        ...provided,
        backgroundColor: state.isFocused ? '#374151' : '#1F2937',
        color: 'white',
    }),
    multiValue: (provided) => ({
        ...provided,
        backgroundColor: '#4B5563',
    }),
    multiValueLabel: (provided) => ({
        ...provided,
        color: 'white',
    }),
};

// Test score ranges for Pakistani context
const testScoreRanges = {
    SAT: { min: 400, max: 1600, typical: 1200 },
    ACT: { min: 1, max: 36, typical: 24 },
    IELTS: { min: 0, max: 9, typical: 6.5 },
    TOEFL: { min: 0, max: 120, typical: 79 },
    GRE: { min: 260, max: 340, typical: 310 }
};

const inputStyle = "w-full bg-gray-100 border border-gray-300 rounded-md shadow-sm p-2 focus:ring-blue-500 focus:border-blue-500 text-black";
const selectStyle = "w-1/3 bg-gray-100 border border-gray-300 rounded-md shadow-sm p-2 focus:ring-blue-500 focus:border-blue-500 text-black";

const LeftPanel = ({ formState, setFormState, onSubmit, loading }) => {
    const handleInputChange = (field, value) => {
        setFormState(prevState => {
            const newState = { ...prevState, [field]: value };

            // Reset test type when switching application type
            if (field === 'applicationType') {
                const availableTests = value === 'undergraduate'
                    ? ['SAT', 'ACT', 'IELTS', 'TOEFL']
                    : ['GRE', 'IELTS', 'TOEFL'];

                // If current test type is not available for new application type, reset to first available
                if (!availableTests.includes(prevState.testType)) {
                    newState.testType = availableTests[0];
                    newState.testScore = testScoreRanges[availableTests[0]]?.typical || '';
                }

                // Reset education level when switching
                newState.educationLevel = value === 'undergraduate' ? 'High School' : 'Bachelor\'s Degree';
            }

            return newState;
        });
    };

    const handlePriorityChange = (name, value) => {
        const priorities = formState.priorities;
        const currentValue = priorities[name];
        const diff = value - currentValue;

        let remainingDiff = diff;
        const otherKeys = Object.keys(priorities).filter(k => k !== name);
        const newPriorities = { ...priorities, [name]: value };

        let attempts = 0;
        while (Math.abs(remainingDiff) > 0.1 && attempts < 10) {
            const distributableKeys = otherKeys.filter(k => (remainingDiff < 0 ? newPriorities[k] < 100 : newPriorities[k] > 0));
            if (distributableKeys.length === 0) break;

            const changePerKey = remainingDiff / distributableKeys.length;

            for (const key of distributableKeys) {
                const newKeyValue = newPriorities[key] - changePerKey;
                if (newKeyValue >= 0 && newKeyValue <= 100) {
                    newPriorities[key] = newKeyValue;
                    remainingDiff -= changePerKey;
                }
            }
            attempts++;
        }

        const total = Object.values(newPriorities).reduce((sum, v) => sum + v, 0);
        if (total !== 100) {
            const lastKey = otherKeys[0];
            newPriorities[lastKey] += 100 - total;
        }

        handleInputChange('priorities', newPriorities);
    };



    // Get available test types based on application type
    const getAvailableTestTypes = () => {
        if (formState.applicationType === 'undergraduate') {
            return ['SAT', 'ACT', 'IELTS', 'TOEFL'];
        } else {
            return ['GRE', 'IELTS', 'TOEFL'];
        }
    };

    // Get placeholder for test score based on selected test type
    const getTestScorePlaceholder = () => {
        const range = testScoreRanges[formState.testType];
        return range ? `${range.min}-${range.max} (typical: ${range.typical})` : 'Score';
    };

    return (
        <div className="bg-white rounded-lg shadow-xl border border-gray-200">
            <div className="p-6">
                <h2 className="text-xl font-semibold mb-2 text-black">Student Profile</h2>
                <p className="text-sm text-gray-600 mb-4">
                    Fill in your details to get personalized university recommendations.
                </p>
            </div>

            <form className="space-y-2">
                {/* Application Type Selection */}
                <div className="px-6 py-4 border-b border-white/20">
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                        Applying for
                    </label>
                    <div className="flex space-x-4">
                        <label className="flex items-center">
                            <input
                                type="radio"
                                name="applicationType"
                                value="undergraduate"
                                checked={formState.applicationType === 'undergraduate'}
                                onChange={(e) => handleInputChange('applicationType', e.target.value)}
                                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                            />
                            <span className="ml-2 text-black">Undergraduate</span>
                        </label>
                        <label className="flex items-center">
                            <input
                                type="radio"
                                name="applicationType"
                                value="graduate"
                                checked={formState.applicationType === 'graduate'}
                                onChange={(e) => handleInputChange('applicationType', e.target.value)}
                                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                            />
                            <span className="ml-2 text-black">Graduate</span>
                        </label>
                    </div>
                </div>

                <StaticSection title="Academic Information">
                    <div className="space-y-4">
                        {/* Conditional GPA Field - Only for Graduate */}
                        {formState.applicationType === 'graduate' && (
                            <div>
                                <label htmlFor="gpa" className="block text-sm font-medium text-gray-700 mb-2">
                                    CGPA: <span className="font-bold text-black">{formState.gpa.toFixed(2)}</span>
                                </label>
                                <input
                                    type="range"
                                    id="gpa"
                                    min="0"
                                    max="4"
                                    step="0.01"
                                    value={formState.gpa}
                                    onChange={(e) => handleInputChange('gpa', parseFloat(e.target.value))}
                                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                />
                            </div>
                        )}

                        {/* Test Scores and Education Level - 2 columns on larger screens */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                            {/* Test Scores */}
                            <div>
                                <label htmlFor="test-type" className="block text-sm font-medium text-gray-700 mb-2">
                                    Test Scores
                                </label>
                                <div className="flex space-x-2">
                                    <select
                                        id="test-type"
                                        value={formState.testType}
                                        onChange={(e) => handleInputChange('testType', e.target.value)}
                                        className={selectStyle.replace('w-1/3', 'w-1/3')}
                                    >
                                        {getAvailableTestTypes().map(test => (
                                            <option key={test} value={test}>{test}</option>
                                        ))}
                                    </select>
                                    <input
                                        type="number"
                                        placeholder={getTestScorePlaceholder()}
                                        value={formState.testScore}
                                        onChange={(e) => handleInputChange('testScore', e.target.value)}
                                        className={inputStyle.replace('w-full', 'w-2/3')}
                                    />
                                </div>
                                <p className="text-xs text-gray-400 mt-1">
                                    {formState.testType}: {testScoreRanges[formState.testType]?.min}-{testScoreRanges[formState.testType]?.max} range
                                </p>
                            </div>

                            {/* Current Education Level - Dynamic based on application type */}
                            <div>
                                <label htmlFor="education-level" className="block text-sm font-medium text-gray-700 mb-2">
                                    {formState.applicationType === 'undergraduate' ? 'Current Education Level' : 'Highest Completed Degree'}
                                </label>
                                <select
                                    id="education-level"
                                    value={formState.educationLevel}
                                    onChange={(e) => handleInputChange('educationLevel', e.target.value)}
                                    className={inputStyle}
                                >
                                    {formState.applicationType === 'undergraduate' ? (
                                        <>
                                            <option>High School</option>
                                            <option>Higher Secondary School Certificate (HSSC)</option>
                                            <option>A-Levels</option>
                                            <option>Foundation Program</option>
                                        </>
                                    ) : (
                                        <>
                                            <option>Bachelor's Degree</option>
                                            <option>Master's Degree</option>
                                            <option>PhD</option>
                                        </>
                                    )}
                                </select>
                            </div>
                        </div>

                        {/* Achievements/Qualifications */}
                        <div>
                            <label htmlFor="achievements" className="block text-sm font-medium text-gray-700 mb-2">
                                {formState.applicationType === 'undergraduate' ? 'Achievements / Qualifications' : 'Research Experience / Publications'}
                            </label>
                            <textarea
                                id="achievements"
                                rows="3"
                                placeholder={formState.applicationType === 'undergraduate'
                                    ? "e.g., National Merit Scholar, Olympiad medals, leadership roles"
                                    : "e.g., Research papers published, thesis work, conference presentations"
                                }
                                value={formState.achievements}
                                onChange={(e) => handleInputChange('achievements', e.target.value)}
                                className={inputStyle}
                            ></textarea>
                        </div>
                    </div>
                </StaticSection>

                <StaticSection title="Preferences">
                    <div className="space-y-4">
                        {/* Country Selection - Full width */}
                        <div>
                            <label htmlFor="country" className="block text-sm font-medium text-gray-700 mb-2">
                                Preferred Countries
                            </label>
                            <Select
                                id="country"
                                isMulti
                                options={countryOptions}
                                className="text-white"
                                styles={customStyles}
                                value={formState.countries}
                                onChange={(selectedOptions) => handleInputChange('countries', selectedOptions)}
                            />
                        </div>

                        {/* Budget Range Slider - Full width */}
                        <div>
                            <label htmlFor="budget" className="block text-sm font-medium text-gray-700 mb-2">
                                Budget: <span className="font-bold text-black">${formState.budget.toLocaleString()}</span>
                            </label>
                            <input
                                type="range"
                                id="budget"
                                min="0"
                                max="100000"
                                step="1000"
                                value={formState.budget}
                                onChange={(e) => handleInputChange('budget', parseInt(e.target.value, 10))}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                        </div>

                        {/* Field of Study and Timeline - 2 columns */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Field of Study */}
                            <div>
                                <label htmlFor="field-of-study" className="block text-sm font-medium text-gray-700 mb-2">
                                    Field of Study
                                </label>
                                <input
                                    type="text"
                                    id="field-of-study"
                                    placeholder="e.g., Computer Science, Medicine"
                                    value={formState.fieldOfStudy}
                                    onChange={(e) => handleInputChange('fieldOfStudy', e.target.value)}
                                    className={inputStyle}
                                />
                            </div>

                            {/* Timeline */}
                            <div>
                                <label htmlFor="timeline" className="block text-sm font-medium text-gray-700 mb-2">
                                    Timeline
                                </label>
                                <select id="timeline" value={formState.timeline} onChange={(e) => handleInputChange('timeline', e.target.value)} className={inputStyle}>
                                    <option>Fall 2025</option>
                                    <option>Spring 2026</option>
                                    <option>Fall 2026</option>
                                    <option>Spring 2027</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </StaticSection>

                <StaticSection title="Priorities">
                    <div className="space-y-4">
                        {/* Priority Sliders - 2 columns on larger screens */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {Object.keys(formState.priorities).map(key => (
                                <div key={key}>
                                    <label className="capitalize text-sm font-medium text-gray-700 mb-2 block">
                                        {key.replace('_', ' ')}: <span className="font-bold text-black">{Math.round(formState.priorities[key])}%</span>
                                    </label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="100"
                                        value={formState.priorities[key]}
                                        onChange={(e) => handlePriorityChange(key, parseFloat(e.target.value))}
                                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                    />
                                </div>
                            ))}
                        </div>
                        <button
                            type="button"
                            onClick={() => handleInputChange('priorities', { reputation: 25, cost: 25, location: 25, culture: 25 })}
                            className="mt-4 w-full bg-gray-200 hover:bg-gray-300 text-black font-bold py-2 px-4 rounded"
                        >
                            Reset to Default
                        </button>
                    </div>
                </StaticSection>

                {/* Submit Button */}
                <div className="p-6">
                    <button
                        type="button"
                        onClick={onSubmit}
                        disabled={loading}
                        className={`w-full py-3 px-6 rounded-lg font-semibold text-white transition-all duration-200 ${loading
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 hover:shadow-lg transform hover:scale-105'
                            }`}
                    >
                        {loading ? (
                            <div className="flex items-center justify-center space-x-2">
                                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                <span>Getting Recommendations...</span>
                            </div>
                        ) : (
                            'Get University Recommendations'
                        )}
                    </button>
                </div>
            </form>
        </div>
    );
};

export default LeftPanel; 