# Passive Seismic Analytics & Deep Learning Portfolio
An exploratory data science repository focusing on time-series processing, passive seismology, and deep learning architectures for environmental geophysics and smart city applications. This portfolio serves as technical preparation for scaling up discrete geophone instrumentation methodologies into high-density Distributed Acoustic Sensing (DAS) frameworks.

---

## 📂 Repository Directory Layout
```text
seismic-analytics-portfolio/
├── data/
│   └── .gitkeep              # Placeholder folder for downloaded seismic files
├── images/
│   ├── step1_spectrogram.png # Save your generated spectrogram plot here
│   ├── step2_cross_corr.png  # Save your cross-correlation plot here
│   └── step3_loss_curve.png  # (Optional) Save your training execution outputs here
├── notebooks/
│   ├── 01_obspy_data_query.ipynb
│   ├── 02_ambient_noise_cross_corr.ipynb
│   └── 03_pytorch_seismic_classifier.ipynb
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Technical Implementation Modules & Source Code
### Module 1: Automated Broadband Data Acquisition & Spectral Analysis
This module establishes a connection to the IRIS FDSN web services utilizing ```ObsPy``` to programmatically ingest raw seismic waveforms. The script demeans, detrends, and filters raw streams to isolate urban cultural background noise, visualizing changes via signal spectrograms.

```python
import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt

# 1. Connect to the IRIS Web Service client
client = Client("IRIS")

# 2. Define a time window (e.g., an hour of urban background noise)
starttime = UTCDateTime("2026-01-01T12:00:00")
endtime = starttime + 3600  # 1 hour

# 3. Download data from a real seismic station (Network: IU, Station: ANMO)
st = client.get_waveforms("IU", "ANMO", "00", "BHZ", starttime, endtime)
tr = st[0]  # Get the primary trace

# 4. Preprocess: Demean and Detrend (Standard practice for seismic noise)
tr.detrend("demean")
tr.detrend("linear")

# 5. Apply a bandpass filter targeting typical urban cultural noise (1.0 - 10.0 Hz)
tr.filter("bandpass", freqmin=1.0, freqmax=10.0)

# 6. Plot the waveform and its spectrogram
tr.plot()
tr.spectrogram(log=True, title="Urban Ambient Noise Characterization")
  ```

### Expected Visual Output:

<table align="center" border="0">
  <tr>
    <td align="center">
      <img src="images/step1_waveform.png" alt="Time-series waveform trace of ambient noise" width="380"/>
    </td>
    <td align="center">
      <img src="images/step1_spectrogram.png" alt="Logarithmic power spectral density spectrogram" width="380"/>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <sub>Figure 1: Time-series waveform trace alongside its corresponding logarithmic power spectral density (PSD) spectrogram highlighting steady ambient urban frequencies.</sub>
    </td>
  </tr>
</table>

### Module 2: Ambient Noise Cross-Correlation (Green's Function Extraction)
To replicate subsurface geohazard detection techniques (such as mapping subsidence or sinkholes), this module runs spatial cross-correlations over parallel continuous noise streams. This process evaluates phase shifts and travel time delays between discrete sensor channels along a structural path.

```python
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

#Simulating continuous ambient noise at two different virtual fiber sensors (DAS channels)
np.random.seed(42)
common_ambient_noise = np.random.normal(0, 1, 5000)

#Let Sensor B be further down the road, so it experiences a time delay (phase shift)
sensor_a_data = common_ambient_noise + np.random.normal(0, 0.5, 5000)
sensor_b_data = np.roll(common_ambient_noise, 150) + np.random.normal(0, 0.5, 5000) 

#Performing cross-correlation to find the seismic wave travel time between sensors
correlation = signal.correlate(sensor_b_data, sensor_a_data, mode='same')
lags = signal.correlation_lags(len(sensor_b_data), len(sensor_a_data), mode='same')

#Finding the peak lag time 
max_idx = np.argmax(correlation)
time_delay = lags[max_idx]
print(f"Empirical Wave Travel Time (Lags): {time_delay} samples")

#Plotting the result (stack to image sinkholes/fault lines)
plt.figure(figsize=(10, 4))
plt.plot(lags, correlation, color='teal')
plt.title("Cross-Correlation Function (Green's Function Extraction)")
plt.xlabel("Time Lags")
plt.ylabel("Correlation Strength")
plt.grid(True)
plt.show()
```

<p align="center">
  <img src="images/step2_cross_corr.png" alt="Empirical Green's Function recovery plot showing travel time lag symmetry" width="70%" />
  <br />
  <sub>Figure 2: Empirical Green's Function recovery displaying clear travel time lag symmetry, isolating coherent signal features out of stochastic ambient vibrations.</sub>
</p>

### Module 3: 1D Temporal Convolutional Neural Networks (CNN) Optimization
Dense DAS architectures generate terabytes of high-frequency data daily, requiring automated event picking. This module defines a deep 1D Convolutional Neural Network built with ```PyTorch``` utilizing batch normalization, max-pooling, and adaptive average pooling layers. It runs a multi-epoch optimization loop tracking cross-entropy loss convergence to simulate model verification.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  

#Defining a 1D CNN Architecture suitable for dense DAS arrays or geophone data
class SeismicClassifier(nn.Module):
    def __init__(self):
        super(SeismicClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # to flatten down the time dimension
        )
        self.classifier = nn.Linear(32, 2) # Binary Classification: 0 = Traffic Noise, 1 = Hazard

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#Instantiating dummy data mimicking a continuous wave (Batch Size = 4, Channels = 1, Time Samples = 1000)
dummy_seismic_stream = torch.randn(4, 1, 1000)
labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

model = SeismicClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # Bumped learning rate slightly for faster toy convergence

#Simulating a multi-epoch training loop to gather historical data for the curve
num_epochs = 30
loss_history = []

print("Starting training loop...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(dummy_seismic_stream)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # to store the loss scalar value
    loss_history.append(loss.item())
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Current Loss: {loss.item():.4f}")

print("\nTraining Complete. Generating loss curve...")

#Plotting and saving the execution outputs
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='#1f77b4', linewidth=2, label='Training Loss')
plt.title('Seismic Classifier Optimization Progress')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

plt.savefig('step3_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("File successfully saved as 'step3_loss_curve.png'")
```

SeismicClassifier Flow Diagram:

```SeismicClassifier Flow Diagram:<br />
Input [Batch, 1, 1000] ──> Conv1D (k=15, s=2) ──> BatchNorm ──> ReLU ──> MaxPool<br />
                       ──> Conv1D (k=7, s=2)  ──> BatchNorm ──> ReLU ──> AdaptiveAvgPool<br />
                       ──> Linear Layer ──────> Output [Batch, 2] (Class Logits)
```

<p align="center">
  <img src="images/step3_loss_curve.png" alt="Multi-epoch gradient descent minimization profile" width="70%" />
  <br />
  <sub>Figure 3: Multi-epoch gradient descent minimization profile demonstrating mathematical stability during spatial waveform feature categorization.</sub>
</p>

## 🛠️ Environment Prerequisites & Setup
To execute the scripts locally, configure your Python virtual environment using the following dependencies:

### Create and activate environment
conda create -n seismic_env python=3.10 -y
conda activate seismic_env

### Install Seismology & ML frameworks
conda install -c conda-forge obspy -y
pip install torch torchvision torchaudio
pip install scipy matplotlib numpy pandas

## 📬 Contact & Affiliation
**Khem Raj Shah**  
*Civil Engineering Researcher | Lead Researcher at [Kathmandu Geo Lab](https://ktmgeolab.org)*  

* **Email:** [kraaj.shah@gmail.com](mailto:kraaj.shah@gmail.com)  
* **GitHub:** [github.com/kraajshah](https://github.com/kraajshah)
