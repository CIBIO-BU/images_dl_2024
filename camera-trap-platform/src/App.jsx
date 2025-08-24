import React, { useState, useEffect } from "react";
import axios from "axios";

// Main App Component with Router
function App() {
  return <MainApp />;
}

// Main Application Component
function MainApp() {
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [detections, setDetections] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [jsonData, setJsonData] = useState(null);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [currentFeedback, setCurrentFeedback] = useState({});
  const [notification, setNotification] = useState(null);

  const handleImageUpload = async (files) => {
    setIsLoading(true);
    setNotification(null);

    const newImages = [];
    const newDetections = {};

    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await axios.post(
          "http://127.0.0.1:8000/api/predict/",
          formData,
          {
            headers: { 
              "Content-Type": "multipart/form-data",
              "Authorization": `Bearer ${localStorage.getItem("token")}`
            },
          }
        );

        const mappedDetections = response.data.detections.map((det) => ({
          label: det.species_prediction,
          confidence: det.species_confidence,
          bbox: det.bbox,
          index: det.index || response.data.detections.indexOf(det),
        }));

        const fileUrl = URL.createObjectURL(file);
        newImages.push({ file, url: fileUrl, name: file.name });
        newDetections[fileUrl] = mappedDetections;
      }

      setImages((prevImages) => [...prevImages, ...newImages]);
      setDetections((prevDetections) => ({ ...prevDetections, ...newDetections }));
      setJsonData((prevJson) => ({ ...prevJson, ...newDetections }));

      if (newImages.length > 0) {
        setSelectedImage(newImages[0].url);
      }

      showNotification("Images processed successfully!", "success");
    } catch (error) {
      console.error("Error uploading image:", error);
      showNotification("Failed to process images", "error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedbackSubmit = async () => {
  if (!selectedImage || Object.keys(currentFeedback).length === 0) {
    showNotification("No feedback to submit", "warning");
    return;
  }

  try {
    // Prepare the feedback payload
    const payload = {
      image_url: selectedImage,
      detections: detections[selectedImage].map(det => ({
        bbox: det.bbox,
        label: det.label,
        confidence: det.confidence,
        index: det.index
      })),
      user_feedback: currentFeedback
    };

    const response = await axios.post(
      "http://127.0.0.1:8000/api/feedback/",
      payload,
      {
        headers: {
          "Content-Type": "application/json",
        }
      }
    );

    if (response.data.status === "success") {
      showNotification("Feedback submitted successfully!", "success");
      setFeedbackOpen(false);
      setCurrentFeedback({});
      
      // Update local state with the feedback
      setDetections(prev => {
        const updated = {...prev};
        if (updated[selectedImage]) {
          updated[selectedImage] = updated[selectedImage].map(det => ({
            ...det,
            ...currentFeedback[det.index]
          }));
        }
        return updated;
      });
    } else {
      throw new Error(response.data.error || "Failed to save feedback");
    }
  } catch (error) {
    console.error("Feedback submission error:", error);
    showNotification(
      error.response?.data?.error || 
      error.message || 
      "Failed to submit feedback",
      "error"
    );
  }
};

// Helper function for downloading JSON
function handleDownloadJson() {
  if (!jsonData) return;
  const blob = new Blob([JSON.stringify(jsonData, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "detections.json";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

  const showNotification = (message, type) => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  };

  useEffect(() => {
    if (feedbackOpen && selectedImage) {
      const initialFeedback = {};
      detections[selectedImage]?.forEach((detection) => {
        initialFeedback[detection.index] = {
          bbox_correct: true,
          species_correct: true,
          correct_species: "",
        };
      });
      setCurrentFeedback(initialFeedback);
    }
  }, [feedbackOpen, selectedImage, detections]);

  return (
    <div style={styles.container}>
      {notification && (
        <div style={{
          ...styles.notification,
          backgroundColor: notification.type === "success" ? "#2ed573" : 
                         notification.type === "error" ? "#ff4757" : "#FFA502",
        }}>
          {notification.message}
        </div>
      )}

      <div style={styles.header}>
        <h1 style={styles.title}>Wildlife Image Classifier</h1>
      </div>

      <div style={styles.buttonRow}>
        <label htmlFor="fileInput" style={styles.uploadButton}>
          {isLoading ? "Processing..." : "Upload Images"}
        </label>
        <input
          id="fileInput"
          type="file"
          accept="image/*"
          onChange={(e) => handleImageUpload(Array.from(e.target.files))}
          disabled={isLoading}
          style={{ display: "none" }}
          multiple
        />
        {jsonData && (
          <>
            <button
              onClick={handleDownloadJson}
              style={styles.downloadButton}
              disabled={isLoading}
            >
              Download JSON
            </button>
            <button
              onClick={() => setFeedbackOpen(true)}
              style={styles.feedbackButton}
              disabled={!selectedImage || isLoading}
            >
              Provide Feedback
            </button>
          </>
        )}
      </div>

      <div style={styles.displayArea}>
        {selectedImage && (
          <div style={styles.imageContainer}>
            <img
              src={selectedImage}
              alt="Selected"
              style={styles.image}
              onLoad={(e) => {
                const { width, height } = e.target;
                setImageSize({ width, height });
              }}
            />
            {detections[selectedImage]?.map((detection, idx) => {
              const [x_min, y_min, x_max, y_max] = detection.bbox;
              const boxWidth = x_max - x_min;
              const boxHeight = y_max - y_min;

              return (
                <div
                  key={idx}
                  style={{
                    ...styles.detectionBox,
                    left: `${(x_min / imageSize.width) * 100}%`,
                    top: `${(y_min / imageSize.height) * 100}%`,
                    width: `${(boxWidth / imageSize.width) * 100}%`,
                    height: `${(boxHeight / imageSize.height) * 100}%`,
                  }}
                >
                  <span style={styles.detectionLabel}>
                    {detection.label} ({Math.round(detection.confidence * 100)}%)
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div style={styles.scrollContainer}>
        {images.map((image, index) => (
          <div
            key={index}
            style={{
              ...styles.thumbnail,
              border: selectedImage === image.url ? "3px solid #1e90ff" : 
                     detections[image.url]?.length > 0 ? "2px solid #1e90ff" : "2px solid #ff4757",
            }}
            onClick={() => setSelectedImage(image.url)}
          >
            <img
              src={image.url}
              alt={`Thumbnail ${index}`}
              style={styles.thumbnailImage}
            />
            {detections[image.url]?.length === 0 && (
              <div style={styles.noAnimalsLabel}>No Animals</div>
            )}
          </div>
        ))}
      </div>

      {feedbackOpen && selectedImage && (
        <div style={styles.feedbackModal}>
          <div style={styles.feedbackContent}>
            <h3 style={styles.feedbackTitle}>Provide Feedback on Predictions</h3>
            <p style={styles.feedbackSubtitle}>Please correct any mistakes in the detections:</p>
            
            {detections[selectedImage]?.map((detection) => (
              <div key={detection.index} style={styles.feedbackItem}>
                <h4 style={styles.detectionHeader}>Detection #{detection.index + 1}</h4>
                <div style={styles.feedbackRow}>
                  <label style={styles.feedbackLabel}>
                    <input
                      type="checkbox"
                      checked={currentFeedback[detection.index]?.bbox_correct ?? true}
                      onChange={(e) =>
                        setCurrentFeedback((prev) => ({
                          ...prev,
                          [detection.index]: {
                            ...prev[detection.index],
                            bbox_correct: e.target.checked,
                          },
                        }))
                      }
                      style={styles.feedbackCheckbox}
                    />
                    Bounding box is correct
                  </label>
                </div>
                <div style={styles.feedbackRow}>
                  <label style={styles.feedbackLabel}>
                    <input
                      type="checkbox"
                      checked={currentFeedback[detection.index]?.species_correct ?? true}
                      onChange={(e) =>
                        setCurrentFeedback((prev) => ({
                          ...prev,
                          [detection.index]: {
                            ...prev[detection.index],
                            species_correct: e.target.checked,
                          },
                        }))
                      }
                      style={styles.feedbackCheckbox}
                    />
                    Species identification is correct
                  </label>
                </div>
                {currentFeedback[detection.index]?.species_correct === false && (
                  <div style={styles.feedbackRow}>
                    <label style={styles.feedbackLabel}>
                      Correct species:
                      <input
                        type="text"
                        value={currentFeedback[detection.index]?.correct_species || ""}
                        onChange={(e) =>
                          setCurrentFeedback((prev) => ({
                            ...prev,
                            [detection.index]: {
                              ...prev[detection.index],
                              correct_species: e.target.value,
                            },
                          }))
                        }
                        style={styles.speciesInput}
                        placeholder="Enter correct species name"
                      />
                    </label>
                  </div>
                )}
              </div>
            ))}

            <div style={styles.feedbackActions}>
              <button
                onClick={() => setFeedbackOpen(false)}
                style={styles.cancelButton}
              >
                Cancel
              </button>
              <button
                onClick={handleFeedbackSubmit}
                style={styles.submitButton}
                disabled={Object.keys(currentFeedback).length === 0}
              >
                Submit Feedback
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  // Authentication styles
  authContainer: {
    maxWidth: "400px",
    margin: "50px auto",
    padding: "30px",
    borderRadius: "10px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    backgroundColor: "#ffffff",
  },
  authTitle: {
    fontSize: "24px",
    fontWeight: "600",
    color: "#2f3542",
    marginBottom: "20px",
    textAlign: "center",
  },
  authForm: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },
  formGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  formLabel: {
    fontSize: "14px",
    fontWeight: "500",
    color: "#57606f",
  },
  formInput: {
    padding: "12px 15px",
    borderRadius: "6px",
    border: "1px solid #dfe6e9",
    fontSize: "14px",
    transition: "border-color 0.3s ease",
    ":focus": {
      outline: "none",
      borderColor: "#1e90ff",
    },
  },
  authButton: {
    padding: "12px",
    backgroundColor: "#1e90ff",
    color: "white",
    border: "none",
    borderRadius: "6px",
    fontSize: "16px",
    fontWeight: "500",
    cursor: "pointer",
    transition: "all 0.3s ease",
    ":hover": {
      backgroundColor: "#187bcd",
      transform: "translateY(-2px)",
    },
  },
  toggleAuthButton: {
    marginTop: "15px",
    background: "none",
    border: "none",
    color: "#1e90ff",
    cursor: "pointer",
    fontSize: "14px",
    textDecoration: "underline",
    transition: "color 0.2s ease",
    ":hover": {
      color: "#187bcd",
    },
  },
  error: {
    color: "#ff4757",
    backgroundColor: "rgba(255, 71, 87, 0.1)",
    padding: "10px 15px",
    borderRadius: "6px",
    marginBottom: "20px",
    fontSize: "14px",
  },
  loading: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    fontSize: "18px",
    color: "#2f3542",
  },

  // Main app styles
  container: {
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "20px",
    fontFamily: "'Arial', sans-serif",
    textAlign: "center",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "30px",
    paddingBottom: "15px",
    borderBottom: "1px solid #dfe6e9",
  },
  title: {
    fontSize: "32px",
    fontWeight: "bold",
    color: "#2f3542",
    margin: 0,
  },
  logoutButton: {
    padding: "8px 16px",
    backgroundColor: "#ff4757",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: "500",
    transition: "all 0.3s ease",
    ":hover": {
      backgroundColor: "#e84118",
      transform: "translateY(-2px)",
    },
  },
  buttonRow: {
    display: "flex",
    justifyContent: "center",
    gap: "15px",
    marginBottom: "25px",
    flexWrap: "wrap",
  },
  uploadButton: {
    padding: "12px 24px",
    backgroundColor: "#1e90ff",
    color: "#fff",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "16px",
    fontWeight: "500",
    transition: "all 0.3s ease",
    ":hover": {
      backgroundColor: "#187bcd",
      transform: "translateY(-2px)",
    },
  },
  downloadButton: {
    padding: "12px 24px",
    backgroundColor: "#2ed573",
    color: "#fff",
    borderRadius: "6px",
    fontSize: "16px",
    fontWeight: "500",
    border: "none",
    cursor: "pointer",
    transition: "all 0.3s ease",
    ":hover": {
      backgroundColor: "#25b864",
      transform: "translateY(-2px)",
    },
    ":disabled": {
      backgroundColor: "#cccccc",
      cursor: "not-allowed",
      transform: "none",
    },
  },
  feedbackButton: {
    padding: "12px 24px",
    backgroundColor: "#FFA502",
    color: "#fff",
    borderRadius: "6px",
    fontSize: "16px",
    fontWeight: "500",
    border: "none",
    cursor: "pointer",
    transition: "all 0.3s ease",
    ":hover": {
      backgroundColor: "#e69500",
      transform: "translateY(-2px)",
    },
    ":disabled": {
      backgroundColor: "#cccccc",
      cursor: "not-allowed",
      transform: "none",
    },
  },
  displayArea: {
    minHeight: "500px",
    position: "relative",
    marginBottom: "20px",
    backgroundColor: "#f5f6fa",
    borderRadius: "10px",
    padding: "20px",
    boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
  },
  imageContainer: {
    position: "relative",
    display: "inline-block",
    maxWidth: "100%",
  },
  image: {
    maxWidth: "100%",
    maxHeight: "500px",
    borderRadius: "8px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
  },
  detectionBox: {
    position: "absolute",
    border: "2px solid #00B894",
    borderRadius: "4px",
    background: "rgba(0, 184, 148, 0.15)",
    boxSizing: "border-box",
  },
  detectionLabel: {
    position: "absolute",
    top: "-25px",
    left: "0",
    background: "#00B894",
    color: "#fff",
    padding: "4px 8px",
    fontSize: "12px",
    fontWeight: "bold",
    borderRadius: "4px",
    whiteSpace: "nowrap",
  },
  scrollContainer: {
    display: "flex",
    overflowX: "auto",
    gap: "15px",
    padding: "15px 0",
    marginTop: "10px",
  },
  thumbnail: {
    position: "relative",
    flex: "0 0 auto",
    borderRadius: "6px",
    cursor: "pointer",
    transition: "all 0.3s ease",
    ":hover": {
      transform: "scale(1.05)",
      boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
    },
  },
  thumbnailImage: {
    width: "120px",
    height: "120px",
    borderRadius: "6px",
    objectFit: "cover",
  },
  noAnimalsLabel: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    background: "rgba(255, 71, 87, 0.8)",
    color: "#fff",
    padding: "4px 8px",
    borderRadius: "4px",
    fontSize: "12px",
    fontWeight: "bold",
  },
  notification: {
    position: "fixed",
    top: "20px",
    left: "50%",
    transform: "translateX(-50%)",
    padding: "10px 20px",
    borderRadius: "4px",
    color: "white",
    zIndex: 1001,
    boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
  },

  // Feedback modal styles
  feedbackModal: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 1000,
    backdropFilter: "blur(5px)",
  },
  feedbackContent: {
    backgroundColor: "white",
    padding: "25px",
    borderRadius: "10px",
    width: "90%",
    maxWidth: "600px",
    maxHeight: "80vh",
    overflowY: "auto",
    boxShadow: "0 5px 15px rgba(0,0,0,0.3)",
  },
  feedbackTitle: {
    fontSize: "20px",
    fontWeight: "600",
    color: "#2f3542",
    marginBottom: "5px",
  },
  feedbackSubtitle: {
    fontSize: "14px",
    color: "#57606f",
    marginBottom: "20px",
  },
  feedbackItem: {
    margin: "20px 0",
    padding: "15px",
    border: "1px solid #dfe6e9",
    borderRadius: "8px",
    backgroundColor: "#f8f9fa",
  },
  feedbackRow: {
    marginBottom: "12px",
  },
  feedbackLabel: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    cursor: "pointer",
    fontSize: "14px",
  },
  feedbackCheckbox: {
    width: "16px",
    height: "16px",
    cursor: "pointer",
  },
  speciesInput: {
    flex: 1,
    padding: "8px 12px",
    borderRadius: "4px",
    border: "1px solid #b2bec3",
    fontSize: "14px",
    marginLeft: "10px",
    transition: "border-color 0.3s ease",
    ":focus": {
      outline: "none",
      borderColor: "#1e90ff",
    },
  },
  feedbackActions: {
    display: "flex",
    justifyContent: "flex-end",
    gap: "10px",
    marginTop: "20px",
  },
  submitButton: {
    padding: "10px 20px",
    backgroundColor: "#2ed573",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontWeight: "500",
    transition: "all 0.2s ease",
    ":hover": {
      backgroundColor: "#25b864",
      transform: "translateY(-2px)",
    },
  },
  cancelButton: {
    padding: "10px 20px",
    backgroundColor: "#ff4757",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontWeight: "500",
    transition: "all 0.2s ease",
    ":hover": {
      backgroundColor: "#e84118",
      transform: "translateY(-2px)",
    },
  },
  detectionHeader: {
    fontSize: "16px",
    color: "#2f3542",
    marginTop: "0",
    marginBottom: "15px",
  },
};

export default App;