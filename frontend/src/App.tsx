import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0])
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first!')
      return
    }

    setUploading(true)
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post('http://localhost:8000/submit', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setResult(JSON.stringify(response.data, null, 2))
    } catch (error) {
      if (axios.isAxiosError(error)) {
        console.error('Error uploading file:', error.response?.data || error.message)
        setResult(`Error: ${error.response?.data?.detail || error.message}`)
      } else {
        console.error('Error uploading file:', error)
        setResult('Error uploading file')
      }
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="container">
      <h1>Image Caption Generator</h1>
      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="file-input"
        />
        <button 
          onClick={handleUpload}
          disabled={!selectedFile || uploading}
          className="upload-button"
        >
          {uploading ? 'Generating Caption...' : 'Generate Caption'}
        </button>
      </div>
      {selectedFile && (
        <div className="preview">
          <p>Selected file: {selectedFile.name}</p>
          <img 
            src={URL.createObjectURL(selectedFile)} 
            alt="Preview" 
            style={{ maxWidth: '300px' }} 
          />
        </div>
      )}
      {result && (
        <div className="result">
          <h3>Generated Caption:</h3>
          <pre>{result}</pre>
        </div>
      )}
    </div>
  )
}

export default App
