const StaticSection = ({ title, children }) => {
    return (
        <div className="border-b border-white/20">
            <div className="px-6 py-4">
                <h3 className="text-lg font-medium text-black mb-4">{title}</h3>
                {children}
            </div>
        </div>
    );
};

export default StaticSection; 