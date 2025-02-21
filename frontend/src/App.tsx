import { useState } from 'react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [captions, setCaptions] = useState<string[]>([])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0])
      setCaptions([])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first!')
      return
    }

    setUploading(true)
    setCaptions([])
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('http://localhost:8000/submit', {
        method: 'POST',
        body: formData,
      })

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader available')

      // Read the stream
      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        // Decode the received chunks and split by newlines
        const text = new TextDecoder().decode(value)
        const lines = text.split('\n').filter(line => line.trim())

        // Process each line individually
        for (const line of lines) {
          try {
            const { caption } = JSON.parse(line)
            // Use a callback function with the previous state to ensure we're always appending to the latest state
            setCaptions(prevCaptions => [...prevCaptions, caption])
          } catch (e) {
            console.error('Error parsing caption:', e)
          }
        }
      }
    } catch (error) {
      console.error('Error uploading file:', error)
      setCaptions(prev => [...prev, `Error: ${error instanceof Error ? error.message : 'Unknown error'}`])
    } finally {
      setUploading(false)
    }
  }

  return (
    <div style={{display: 'flex', flexDirection: 'row'}}>
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
            {uploading ? 'Generating Captions...' : 'Generate Captions'}
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
        
      </div>
      {captions.length > 0 && (
          <div className="result" style={{alignSelf: 'center'}}>
            <h3>Generated Captions:</h3>
            <ul className="captions-list">
              {captions.map((caption, index) => (
                <li key={index} className="caption-item">
                  {caption}
                </li>
              ))}
            </ul>
          </div>
        )}
    </div>
  )
}

export default App
