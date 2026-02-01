import './style.css'

const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');
const annotatedImage = document.getElementById('annotatedImage');
const totalCount = document.getElementById('totalCount');
const resetBtn = document.getElementById('resetBtn');

// Stats elements
const stats = {
  healthy: {
    count: document.getElementById('healthyCount'),
    percent: document.getElementById('healthyPercent'),
    bar: document.getElementById('healthyBar')
  },
  black: {
    count: document.getElementById('blackCount'),
    percent: document.getElementById('blackPercent'),
    bar: document.getElementById('blackBar')
  },
  rain_damaged: {
    count: document.getElementById('rainCount'),
    percent: document.getElementById('rainPercent'),
    bar: document.getElementById('rainBar')
  }
};

// Event Listeners
dropZone.addEventListener('click', () => imageInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('active');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    processImage(file);
  }
});

imageInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    processImage(file);
  }
});

resetBtn.addEventListener('click', () => {
  resultsSection.classList.remove('show');
  setTimeout(() => {
    resultsSection.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
  }, 300);
});

async function processImage(file) {
  // Show loading
  loadingOverlay.style.display = 'flex';
  
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://localhost:8000/detect', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Detection failed');

    const data = await response.json();
    displayResults(data);
  } catch (error) {
    console.error(error);
    alert('Error processing image. Make sure the backend server is running.');
  } finally {
    loadingOverlay.style.display = 'none';
  }
}

function displayResults(data) {
  // Update image
  annotatedImage.src = data.annotated_image;
  
  // Update total count
  totalCount.innerText = data.total_seeds;

  // Update stats
  Object.keys(stats).forEach(key => {
    const apiData = data.stats[key] || { count: 0, percentage: 0 };
    const elements = stats[key];
    
    // Animate numbers
    animateValue(elements.count, 0, apiData.count, 500);
    elements.percent.innerText = `${apiData.percentage}%`;
    
    // Animate bar
    setTimeout(() => {
      elements.bar.style.width = `${apiData.percentage}%`;
    }, 100);
  });

  // Show results section
  document.querySelector('.upload-section').style.display = 'none';
  resultsSection.style.display = 'grid';
  setTimeout(() => {
    resultsSection.classList.add('show');
  }, 50);
}

function animateValue(obj, start, end, duration) {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    obj.innerHTML = Math.floor(progress * (end - start) + start);
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}
