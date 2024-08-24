import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
import mne
from scipy.signal import butter, lfilter
from mne.filter import notch_filter
import matplotlib.pyplot as plt

# Function for bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Load the dataset
data = pd.read_csv("D:/MINI/EEG_Eye_State_Classification.csv/EEG_Eye_State_Classification.csv")

# Assuming the last column is the target and the rest are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Define the sample rate and the low and high cutoff frequencies
fs = 128.0
lowcut = 0.5
highcut = 50.0

# Apply the bandpass filter
X = butter_bandpass_filter(X, lowcut, highcut, fs, order=6)

# Define the power line frequency
freq = 60.0

# Apply the notch filter
X = notch_filter(X, fs, freq)

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert them to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# Create Tensor datasets
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# Define a batch size
batch_size = 512

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Define the ANN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization after first layer
        self.dp1 = nn.Dropout(0.01)  # Dropout after first layer
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  # Batch normalization after second layer
        self.dp2 = nn.Dropout(0.01)  # Dropout after second layer
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)  # Batch normalization after third layer
        self.dp3 = nn.Dropout(0.01)  # Dropout after third layer
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)  # Batch normalization after fourth layer
        self.dp4 = nn.Dropout(0.01)  # Dropout after fourth layer
        self.fc5 = nn.Linear(64, 2)  # 2 output classes: open, closed

    def forward(self, x):
        x = self.dp1(self.bn1(torch.tanh(self.fc1(x))))
        x = self.dp2(self.bn2(torch.tanh(self.fc2(x)))) 
        x = self.dp3(self.bn3(torch.tanh(self.fc3(x))))
        x = self.dp4(self.bn4(torch.tanh(self.fc4(x))))
        x = torch.sigmoid(self.fc5(x))
        return x

# Instantiate the model
model = Net()

# Calculate class weights
class_weights = torch.tensor([1.0, y_train[y_train==0].size(0) / y_train[y_train==1].size(0)])

# Define loss function (CrossEntropyLoss for multi-class classification)
criterion = nn.CrossEntropyLoss(weight=class_weights)


# Define loss function (CrossEntropyLoss for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 1000
for epoch in range(epochs):
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

# Test the model
from sklearn.metrics import confusion_matrix

total=0
correct=0

# Initialize lists to store true and predicted labels
true_labels = []
pred_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Append current true and predicted labels to lists
        true_labels.extend(labels.tolist())
        pred_labels.extend(predicted.tolist())

print('Accuracy: %.2f %%' % (100 * correct / total))

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Assuming that `true_labels` are the true labels and `pred_labels` are the predicted labels
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, pred_labels)


print('Precision: %.2f' % precision)
print('Recall: %.2f' % recall)
print('F1 Score: %.2f' % f1)
print('AUC-ROC: %.2f' % roc_auc)

# Create confusion matrix
conf_mat = confusion_matrix(true_labels, pred_labels)
print('Confusion Matrix:\n', conf_mat)

import matplotlib.pyplot as plt
import seaborn as sns

# Create confusion matrixc
conf_mat = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()