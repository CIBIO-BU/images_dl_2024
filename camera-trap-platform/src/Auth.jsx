import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const Auth = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: "",
    password: "",
    email: "",
  });
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    try {
      const endpoint = isLogin ? "/api/login/" : "/api/register/";
      const response = await axios.post(
        `http://127.0.0.1:8000${endpoint}`,
        formData
      );

      if (isLogin) {
        localStorage.setItem("token", response.data.token);
        onLogin(response.data.user);
        navigate("/");
      } else {
        setIsLogin(true); // Switch to login after successful registration
      }
    } catch (err) {
      setError(
        err.response?.data?.message || 
        (isLogin ? "Login failed" : "Registration failed")
      );
    }
  };

  return (
    <div style={styles.authContainer}>
      <h2>{isLogin ? "Login" : "Sign Up"}</h2>
      {error && <div style={styles.error}>{error}</div>}
      <form onSubmit={handleSubmit} style={styles.form}>
        {!isLogin && (
          <div style={styles.formGroup}>
            <label>Email:</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>
        )}
        <div style={styles.formGroup}>
          <label>Username:</label>
          <input
            type="text"
            name="username"
            value={formData.username}
            onChange={handleChange}
            required
          />
        </div>
        <div style={styles.formGroup}>
          <label>Password:</label>
          <input
            type="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
          />
        </div>
        <button type="submit" style={styles.submitButton}>
          {isLogin ? "Login" : "Sign Up"}
        </button>
      </form>
      <button
        onClick={() => setIsLogin(!isLogin)}
        style={styles.toggleButton}
      >
        {isLogin ? "Need an account? Sign Up" : "Already have an account? Login"}
      </button>
    </div>
  );
};

const styles = {
  authContainer: {
    maxWidth: "400px",
    margin: "50px auto",
    padding: "20px",
    borderRadius: "8px",
    boxShadow: "0 0 10px rgba(0,0,0,0.1)",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "15px",
  },
  formGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "5px",
  },
  error: {
    color: "red",
    marginBottom: "15px",
  },
  submitButton: {
    padding: "10px",
    backgroundColor: "#1e90ff",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
  },
  toggleButton: {
    marginTop: "15px",
    background: "none",
    border: "none",
    color: "#1e90ff",
    cursor: "pointer",
    textDecoration: "underline",
  },
};

export default Auth;