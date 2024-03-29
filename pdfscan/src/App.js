import React, { useState, useRef } from 'react';
import './App.css';
import logo from './images/dsfsf.png'; // Import the logo image
import FileForm from './components/FileForm'; // Import the FileForm component

const UploadButton = ({ fileInput }) => {
  return (
    <div>
      <FileForm />
    </div>
  );
}

const MessageBox = ({ message, handleMessageChange, handleSendMessage }) => {
  return (
    <div className="message-box">
      <input 
        type="text" 
        placeholder="Send Query" 
        className="message-input" 
        value={message} 
        onChange={handleMessageChange} 
      />
      <button 
        className="send-button" 
        onClick={handleSendMessage}
      >
        <div className="send-icon"></div>
      </button>
    </div>
  );
}

const App = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [responseBody, setResponseBody] = useState(""); // State to store response body
  const [showResponse, setShowResponse] = useState(false); // State to control response display
  const [queryText, setQueryText] = useState(""); // State to store query text
  const fileInput = useRef(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  }

  const handleMessageChange = (event) => {
    setMessage(event.target.value);
  }

  const handleSendMessage = async () => {
    try {
      const endpoint = 'http://127.0.0.1:8000/query';
      const formData = new URLSearchParams();
      formData.append('question', message);
  
      // Update query text state
      setQueryText(message);
  
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        setResponseBody(JSON.stringify(data)); // Store response body in state
        setShowResponse(true); // Show the response
        console.log('Query Response:', data);
      } else {
        console.error('Failed to Send Query:', response.statusText);
      }
  
      // Display alert after successful response
      alert('Query Received.Please Wait.');
    } catch (error) {
      console.error('Error sending query:', error);
    }
  }
  
  
  
  return (
    <div className="app-container">
      <div className="header">
        <img src={logo} alt="AiPlanet" className="logo" />
        <UploadButton fileInput={fileInput} />
      </div>
      {queryText && (
        <div className="query-container" style={{ backgroundColor: '#d4edda', padding: '10px' }}>
          <p>User Query: {queryText}</p>
        </div>
      )}
      <div className="response-body" style={{ backgroundColor: showResponse ? '#cce5ff' : 'inherit', padding: '10px' }}>{responseBody}</div> {/* Display response body */}
      <input 
        type="file" 
        accept=".pdf" 
        ref={fileInput} 
        onChange={handleFileChange} 
        style={{ display: 'none' }} 
      />
      <div className="footer">
        <MessageBox 
          message={message} 
          handleMessageChange={handleMessageChange} 
          handleSendMessage={handleSendMessage} 
        />
      </div>
    </div>
  );
}

export default App; 
