import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";

const ImageUploader = ({ onImageUpload }) => {
  const onDrop = useCallback(
    (acceptedFiles) => {
      onImageUpload(acceptedFiles[0]);
    },
    [onImageUpload]
  );

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: "image/*",
  });

  return (
    <div {...getRootProps()} style={dropzoneStyles}>
      <input {...getInputProps()} />
      <p>Drag & drop a camera trap image here, or click to select one.</p>
    </div>
  );
};

const dropzoneStyles = {
  border: "2px dashed #007bff",
  borderRadius: "5px",
  padding: "20px",
  textAlign: "center",
  cursor: "pointer",
  margin: "20px 0",
};

export default ImageUploader;