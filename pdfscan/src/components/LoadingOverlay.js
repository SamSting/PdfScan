import React from 'react';
import './LoadingOverlay.css'; // Import CSS for styling loading overlay

// LoadingOverlay component
const LoadingOverlay = () => {
  return (
    <div className="loading-overlay">
      <div className="loading-icon"></div> {/* Add your loading icon or spinner here */}
    </div>
  );
};

export default LoadingOverlay;
