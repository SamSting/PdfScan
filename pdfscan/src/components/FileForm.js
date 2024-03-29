import React, { useState } from 'react';

function FileForm() {
  const [file, setFile] = useState(null);

  const handleFileInputChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleFileUpload = async () => {
    const formData = new FormData();
    formData.append('file_upload', file);

    try {
      const endpoint = 'http://127.0.0.1:8000/uploadfile';
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('File Uploaded Successfully!', file.name);
      } else {
        console.error('Failed to Upload File.');
      }
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h1>Upload</h1>
      <form onSubmit={(e) => e.preventDefault()}>
        <div style={{ marginBottom: '20px' }}>
          <input type="file" onChange={handleFileInputChange} />
        </div>
        <button type="button" onClick={handleFileUpload}>Upload</button>
      </form>
      {file && <p>{file.name}</p>}
    </div>
  );
}

export default FileForm;
