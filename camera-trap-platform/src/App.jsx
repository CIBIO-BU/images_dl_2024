import React, { useState } from "react";
import axios from "axios";

function App() {
  const [images, setImages] = useState([]); // All uploaded images
  const [selectedImage, setSelectedImage] = useState(null); // Currently selected image
  const [detections, setDetections] = useState({}); // Detections for each image
  const [isLoading, setIsLoading] = useState(false);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 }); // Image dimensions

  const handleImageUpload = async (files) => {
    setIsLoading(true);

    const newImages = [];
    const newDetections = {};

    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await axios.post("http://127.0.0.1:8000/api/predict/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        newImages.push({ file, url: URL.createObjectURL(file) });
        newDetections[file.name] = response.data.detections;
      } catch (error) {
        console.error("Error uploading image:", error);
      }
    }

    setImages((prevImages) => [...prevImages, ...newImages]);
    setDetections((prevDetections) => ({ ...prevDetections, ...newDetections }));
    setIsLoading(false);

    // Select the first image by default
    if (newImages.length > 0) {
      setSelectedImage(newImages[0].url);
    }
  };

  const handleImageSelect = (imageUrl) => {
    setSelectedImage(imageUrl);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Wildlife Image Classifier</h1>
      <div style={styles.uploadContainer}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => handleImageUpload(Array.from(e.target.files))}
          disabled={isLoading}
          style={styles.fileInput}
          id="fileInput"
          multiple
        />
        <label htmlFor="fileInput" style={styles.uploadButton}>
          {isLoading ? "Processing..." : "Upload Images"}
        </label>
      </div>
      {isLoading && <p style={styles.loadingText}>Loading...</p>}
      {selectedImage && (
        <div style={styles.imageContainer}>
          <img
            src={selectedImage}
            alt="Selected"
            style={styles.image}
            onLoad={(e) => {
              const imageWidth = e.target.width;
              const imageHeight = e.target.height;
              setImageSize({ width: imageWidth, height: imageHeight });
            }}
          />
          {detections[selectedImage]?.map((detection, index) => {
            const [x_min, y_min, x_max, y_max] = detection.bbox;
            const left = (x_min / imageSize.width) * 100;
            const top = (y_min / imageSize.height) * 100;
            const width = ((x_max - x_min) / imageSize.width) * 100;
            const height = ((y_max - y_min) / imageSize.height) * 100;

            return (
              <div
                key={index}
                style={{
                  position: "absolute",
                  left: `${left}%`,
                  top: `${top}%`,
                  width: `${width}%`,
                  height: `${height}%`,
                  border: "2px solid #ff4757",
                  borderRadius: "4px",
                  boxShadow: "0 0 8px rgba(255, 71, 87, 0.6)",
                }}
              >
                <span style={styles.label}>
                  {detection.label} ({Math.round(detection.confidence * 100)}%)
                </span>
              </div>
            );
          })}
        </div>
      )}
      <div style={styles.scrollContainer}>
        {images.map((image, index) => (
          <div
            key={index}
            style={{
              ...styles.thumbnail,
              border: detections[image.file.name]?.length > 0 ? "2px solid #1e90ff" : "2px solid #ff4757",
            }}
            onClick={() => handleImageSelect(image.url)}
          >
            <img src={image.url} alt={`Thumbnail ${index}`} style={styles.thumbnailImage} />
            {detections[image.file.name]?.length === 0 && (
              <div style={styles.noAnimalsLabel}>No Animals</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "800px",
    margin: "0 auto",
    padding: "20px",
    fontFamily: "'Arial', sans-serif",
    textAlign: "center",
  },
  title: {
    fontSize: "32px",
    fontWeight: "bold",
    color: "#2f3542",
    marginBottom: "20px",
  },
  uploadContainer: {
    marginBottom: "20px",
  },
  fileInput: {
    display: "none",
  },
  uploadButton: {
    display: "inline-block",
    padding: "10px 20px",
    backgroundColor: "#1e90ff",
    color: "#fff",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "16px",
    transition: "background-color 0.3s ease",
  },
  loadingText: {
    fontSize: "18px",
    color: "#2f3542",
  },
  imageContainer: {
    position: "relative",
  },
  image: {
    maxWidth: "100%",
    borderRadius: "8px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
  },
  label: {
    position: "absolute",
    top: "-20px",
    left: "0",
    background: "#ff4757",
    color: "#fff",
    padding: "4px 8px",
    borderRadius: "4px",
    fontSize: "12px",
    fontWeight: "bold",
  },
  scrollContainer: {
    display: "flex",
    overflowX: "auto",
    marginTop: "20px",
    padding: "10px 0",
  },
  thumbnail: {
    position: "relative",
    flex: "0 0 auto",
    marginRight: "10px",
    borderRadius: "4px",
    cursor: "pointer",
    transition: "transform 0.2s ease",
  },
  thumbnailImage: {
    width: "100px",
    height: "100px",
    borderRadius: "4px",
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
};

export default App;