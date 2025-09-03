import "./App.css";

function App() {
  const handleDownload = (platform: string) => {
    // TODO: Implement actual download functionality
    alert(`Download for ${platform} will be available soon!`);
  };

  return (
    <div className="app">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1 className="hero-title">Photo Grouper</h1>
          <p className="hero-subtitle">
            Intelligent photo organization using machine learning
          </p>
          <p className="hero-description">
            Automatically group similar photos, detect duplicates, and organize
            your photo collection with AI-powered similarity detection.
          </p>
          <div className="hero-buttons">
            <button
              className="download-btn primary"
              onClick={() => handleDownload("macOS")}
            >
              Download for macOS
            </button>
            <button
              className="download-btn secondary"
              onClick={() => handleDownload("Windows")}
            >
              Download for Windows
            </button>
            <button
              className="download-btn secondary"
              onClick={() => handleDownload("Linux")}
            >
              Download for Linux
            </button>
          </div>
        </div>
        <div className="hero-image">
          <div className="screenshot-placeholder">
            <span>📸</span>
            <p>App Screenshot</p>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <h2>Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <h3>Smart Photo Grouping</h3>
            <p>
              Automatically groups similar photos using deep learning embeddings
              and cosine similarity
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Real-time Threshold Adjustment</h3>
            <p>
              Dynamically regroup photos by adjusting similarity threshold with
              a slider
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔍</div>
            <h3>Duplicate Detection</h3>
            <p>Identifies and manages duplicate images based on file hashes</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🚀</div>
            <h3>Async Image Loading</h3>
            <p>
              Smooth UI experience with background image loading and priority
              queuing
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📁</div>
            <h3>Multiple Sessions</h3>
            <p>
              Manage different grouping sessions for various photo collections
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📤</div>
            <h3>Export Functionality</h3>
            <p>Export selected groups to organized folders</p>
          </div>
        </div>
      </section>

      {/* System Requirements */}
      <section className="requirements">
        <h2>System Requirements</h2>
        <div className="requirements-grid">
          <div className="requirement-card">
            <h3>🍎 macOS</h3>
            <ul>
              <li>macOS 10.15 or later</li>
              <li>4GB RAM minimum</li>
              <li>500MB free storage</li>
            </ul>
          </div>
          <div className="requirement-card">
            <h3>🪟 Windows</h3>
            <ul>
              <li>Windows 10 or later</li>
              <li>4GB RAM minimum</li>
              <li>500MB free storage</li>
            </ul>
          </div>
          <div className="requirement-card">
            <h3>🐧 Linux</h3>
            <ul>
              <li>Ubuntu 20.04+ / Similar</li>
              <li>4GB RAM minimum</li>
              <li>500MB free storage</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Installation Instructions */}
      <section className="installation">
        <h2>Installation</h2>
        <div className="installation-steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>Download</h3>
              <p>Click the download button for your operating system above</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>Install</h3>
              <p>Run the installer and follow the on-screen instructions</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3>Launch</h3>
              <p>Open Photo Grouper and start organizing your photos!</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p>
          Photo Grouper is open source software licensed under the MIT License.{" "}
          <a
            href="https://github.com/junha6316/photo-grouper"
            target="_blank"
            rel="noopener noreferrer"
          >
            View on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
