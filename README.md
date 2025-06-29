## Medical X-ray Classification Project
This project classifies chest X-rays into COVID-19, Pneumonia, Normal, and Tuberculosis.
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Medical X-ray Classifier</title>
  <link rel="icon" type="image/x-icon" href="{{ url_for('static', filename='favicon.ico') }}">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background-color: #e8f1f5;
      font-family: 'Segoe UI', sans-serif;
    }
    .container {
      margin-top: 60px;
      text-align: center;
    }
    .card {
      padding: 40px;
      border-radius: 20px;
      background-color: #fff;
      box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
    }
    h1 {
      font-size: 2.5rem;
      margin-bottom: 25px;
      color: #0d6efd;
    }
    h4, h5 {
      color: #333;
      font-weight: 600;
    }
    .sample-img {
      max-height: 150px;
      border: 2px solid #dee2e6;
    }
    footer {
      margin-top: 60px;
      background-color: #d6d6d6;
      color: #333;
      text-align: center;
      padding: 15px 0;
      font-size: 15px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>🩻 Medical X-ray Diagnosis</h1>
      <form method="POST" action="/predict" enctype="multipart/form-data">
        <div class="mb-3">
          <input type="file" name="file" accept="image/*" class="form-control" required>
        </div>
        <img id="loading-spinner" src="{{ url_for('static', filename='spinner.gif') }}" style="display:none; width: 60px; margin: 10px;">
        <br>
        <button type="submit" class="btn btn-primary btn-lg">🔍 Predict Disease</button>
      </form>
    </div>

    <div class="mt-5">
      <h4>📈 Model Accuracy & Loss Graphs</h4>
      <div class="row justify-content-center mt-3">
        <div class="col-md-6">
          <img src="{{ url_for('static', filename='accuracy_plot.png') }}" class="img-fluid rounded shadow mb-3">
        </div>
        <div class="col-md-6">
          <img src="{{ url_for('static', filename='loss_plot.png') }}" class="img-fluid rounded shadow mb-3">
        </div>
      </div>
    </div>

    <div class="mt-5 text-start">
      <h4>🧠 Disease Class Information</h4>
      <ul>
        <li><strong>COVID-19:</strong> Viral lung infection with patchy opacities.</li>
        <li><strong>Normal:</strong> No abnormalities in the X-ray image.</li>
        <li><strong>Pneumonia:</strong> Infection with visible white patches.</li>
        <li><strong>Tuberculosis:</strong> Lung cavities or nodules visible.</li>
      </ul>
    </div>

    <div class="mt-5">
      <h4>🖼️ Sample X-ray Gallery</h4>
      <div class="row text-center mt-3">
        <div class="col">
          <p><strong>Normal</strong></p>
          <img src="{{ url_for('static', filename='NORMAL.jpg') }}" class="img-thumbnail sample-img">
        </div>
        <div class="col">
          <p><strong>Pneumonia</strong></p>
          <img src="{{ url_for('static', filename='PNEUMONIA.jpg') }}" class="img-thumbnail sample-img">
        </div>
        <div class="col">
          <p><strong>COVID-19</strong></p>
          <img src="{{ url_for('static', filename='COVID19.jpg') }}" class="img-thumbnail sample-img">
        </div>
        <div class="col">
          <p><strong>Tuberculosis</strong></p>
          <img src="{{ url_for('static', filename='TUBERCULOSIS.jpg') }}" class="img-thumbnail sample-img">
        </div>
      </div>
    </div>
  </div>

  <footer>
    © 2025 Developed by <strong>Momin Aakef</strong>
  </footer>

  <script>
    const form = document.querySelector("form");
    const spinner = document.getElementById("loading-spinner");
    form.addEventListener("submit", () => {
      spinner.style.display = "inline-block";
    });
  </script>
</body>
</html>




<!-- templates/result.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Prediction Result</title>
  <link rel="icon" type="image/x-icon" href="{{ url_for('static', filename='favicon.ico') }}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background-color: #e8f1f5;
      font-family: 'Segoe UI', sans-serif;
    }
    .card {
      margin-top: 40px;
      padding: 30px;
      border-radius: 20px;
      background-color: #fff;
      box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
    }
    .progress-bar {
      transition: width 1.2s ease-in-out;
    }
    .result-badge {
      font-size: 1.2rem;
      padding: 10px 20px;
    }
    footer {
      margin-top: 60px;
      background-color: #d6d6d6;
      color: #333;
      text-align: center;
      padding: 15px 0;
      font-size: 15px;
    }
    .btn-back:hover {
      transform: scale(1.03);
    }
  </style>
</head>
<body>
  <div class="container text-center">
    <div class="card">
      <h2 class="text-success mb-3">✅ Prediction Result</h2>
      <h4><span class="badge bg-primary result-badge">{{ prediction }}</span></h4>

      <!-- Image -->
      <img src="{{ image_path }}" alt="Uploaded X-ray" class="img-fluid mt-4 rounded shadow" style="max-width: 450px;">

      <!-- Confidence -->
      <div class="mt-5 text-start">
        <h5 class="text-center">📊 Confidence Scores:</h5>
        {% for label, score in confidence.items() %}
          <strong>{{ label }}</strong>
          <div class="progress mb-3">
            <div class="progress-bar bg-info" role="progressbar"
                 style="width: {{ score }}%;" aria-valuenow="{{ score }}" aria-valuemin="0" aria-valuemax="100">
              {{ score }}%
            </div>
          </div>
        {% endfor %}
      </div>

      <a href="/" class="btn btn-secondary mt-4 btn-back">← Predict Another Image</a>
    </div>
  </div>

  <footer>
    © 2025 Developed by <strong>Momin Aakef</strong>
  </footer>
</body>
</html>
