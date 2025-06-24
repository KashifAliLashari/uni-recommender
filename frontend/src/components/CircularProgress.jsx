const CircularProgress = ({ percentage }) => {
    const sqSize = 100;
    const strokeWidth = 10;
    const radius = (sqSize - strokeWidth) / 2;
    const viewBox = `0 0 ${sqSize} ${sqSize}`;
    const dashArray = radius * Math.PI * 2;
    const dashOffset = dashArray - (dashArray * (percentage || 0)) / 100;

    const gradientId = "progressGradient";

    return (
        <svg width={sqSize} height={sqSize} viewBox={viewBox}>
            <defs>
                <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#5efeff" />
                    <stop offset="100%" stopColor="#ff7575" />
                </linearGradient>
            </defs>
            <circle
                className="fill-none stroke-gray-200"
                cx={sqSize / 2}
                cy={sqSize / 2}
                r={radius}
                strokeWidth={`${strokeWidth}px`}
            />
            <circle
                className="fill-none"
                cx={sqSize / 2}
                cy={sqSize / 2}
                r={radius}
                strokeWidth={`${strokeWidth}px`}
                transform={`rotate(-90 ${sqSize / 2} ${sqSize / 2})`}
                style={{
                    stroke: `url(#${gradientId})`,
                    strokeDasharray: dashArray,
                    strokeDashoffset: dashOffset,
                    strokeLinecap: 'round',
                    transition: 'stroke-dashoffset 0.3s ease',
                }}
            />
            <text
                className="fill-black font-bold text-xl"
                x="50%"
                y="50%"
                dy=".3em"
                textAnchor="middle"
            >
                {`${Math.round(percentage || 0)}%`}
            </text>
        </svg>
    );
};

export default CircularProgress; 