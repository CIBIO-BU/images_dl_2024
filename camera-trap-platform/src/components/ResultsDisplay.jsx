import React from "react";

const ResultsDisplay = ({ results }) => {
  if (!results) return <p>No results to display</p>;

  return (
    <div style={resultsStyles}>
      <h3>Classification Results</h3>
      <ul>
        {results.map((result, index) => (
          <li key={index}>
            <strong>{result.label}</strong>: {Math.round(result.confidence * 100)}% confidence
          </li>
        ))}
      </ul>
    </div>
  );
};

const resultsStyles = {
  marginTop: "20px",
  padding: "10px",
  border: "1px solid #ddd",
  borderRadius: "5px",
};

export default ResultsDisplay;