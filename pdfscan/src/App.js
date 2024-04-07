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
  const [loading, setLoading] = useState(false); // Loading state
  const fileInput = useRef(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  }

  const handleMessageChange = (event) => {
    setMessage(event.target.value);
  }

  const handleSendMessage = async () => {
    try {
      setLoading(true); // Set loading state to true

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
        const chatHistory = JSON.parse(data.chat_history);
        const aiResponse = chatHistory.find(message => message.type === 'AIMessage');
        if (aiResponse) {
          setResponseBody(aiResponse.content); // Store AI response in state
          setShowResponse(true); // Show the response
          console.log('AI Response:', aiResponse.content);
        } else {
          console.error('No AI response found in chat history');
        }
      } else {
        console.error('Failed to Send Query:', response.statusText);
      }
  
      // Display alert after unsuccessful response
    } catch (error) {
      console.error('Error sending query:', error);
    } finally {
      setLoading(false); // Set loading state to false regardless of success or failure
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
      {loading && ( // Render loading overlay when loading is true
        <div className="loading-overlay">
          <div className="loading-icon"></div>
        </div>
      )}
    </div>
  );
}

export default App;
