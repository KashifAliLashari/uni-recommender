const SkeletonCard = () => (
    <div className="bg-gray-800 p-4 rounded-lg shadow-lg animate-pulse">
        <div className="flex items-center space-x-4">
            <div className="w-16 h-16 bg-gray-700 rounded-full"></div>
            <div className="flex-1 space-y-2">
                <div className="w-3/4 h-4 bg-gray-700 rounded"></div>
                <div className="w-1/2 h-4 bg-gray-700 rounded"></div>
            </div>
        </div>
        <div className="mt-4 space-y-2">
            <div className="w-full h-4 bg-gray-700 rounded"></div>
            <div className="w-5/6 h-4 bg-gray-700 rounded"></div>
        </div>
    </div>
);

export default SkeletonCard; 