// Select DOM elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('preview');
const uploadBtn = document.getElementById('uploadBtn');
const clearBtn = document.getElementById('clearBtn');
const resultDiv = document.getElementById('result');
const heading = document.getElementById('heading');
const dragText = document.getElementById('dragText');
const uploadIcon = document.getElementById('upload-icon');

// ===== DRAG & DROP EVENTS =====
dropArea.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
  dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', (event) => {
  event.preventDefault();
  dropArea.classList.remove('dragover');

  const file = event.dataTransfer.files[0];
  if (file) {
    fileInput.files = event.dataTransfer.files; // store in input
    showPreview(file);
  }
});

// ===== CLICK EVENT: OPEN FILE DIALOG =====
dropArea.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    showPreview(file);
  }
});

// ===== SHOW IMAGE PREVIEW & REMOVE DRAG ICON/TEXT =====
function showPreview(file) {
  // Remove the drag-and-drop text & icon
  dragText.style.display = 'none';
  uploadIcon.style.display = 'none';
  dropArea.style.border = 'none';
  
  // Update heading
  heading.textContent = 'Your Selected Image';

  // Show the image preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block'; // show the preview
  };
  reader.readAsDataURL(file);
}

// ===== UPLOAD BUTTON HANDLER =====
uploadBtn.addEventListener('click', () => {
  const file = fileInput.files[0];
  if (!file) {
    alert('Please select an image first!');
    return;
  }
  uploadImage(file);
});

// ===== CLEAR BUTTON HANDLER =====
clearBtn.addEventListener('click', () => {
  // Reset the heading
  heading.textContent = 'Drag and drop an image of a potato plant leaf to process';

  // Show the drag icon & text
  dragText.style.display = 'block';
  uploadIcon.style.display = 'block';
  dropArea.style.border = '2px dashed #ccc';

  // Hide preview
  previewImg.style.display = 'none';
  previewImg.src = '';

  // Clear result
  resultDiv.innerHTML = '';

  // Reset the file input
  fileInput.value = null;
});

// ===== SEND IMAGE TO FASTAPI =====
function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  // Update URL if your FastAPI is hosted elsewhere
  fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData,
  })
  .then((response) => response.json())
  .then((data) => {
    resultDiv.innerHTML = `
      Predicted Class: ${data.class}<br>
      Confidence: ${(data.confidence * 100).toFixed(2)}%
    `;
  })
  .catch((error) => {
    console.error('Error:', error);
    resultDiv.innerHTML = 'Error in prediction. Please try again.';
  });
}
