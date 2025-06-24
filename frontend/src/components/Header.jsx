import { Wifi } from 'lucide-react';

const Header = () => {
    return (
        <header className="sticky top-0 bg-white shadow-md z-50 border-b border-gray-200">
            <div className="container mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <h1 className="text-2xl font-bold text-black">
                        University Matcher
                    </h1>
                    <div className="flex items-center space-x-2">
                        <span className="text-sm text-green-400">API Status</span>
                        <Wifi className="h-5 w-5 text-green-400" />
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header; 