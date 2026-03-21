# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="832" height="834" alt="image" src="https://github.com/user-attachments/assets/8dedd7e7-ae0d-4ab9-b635-4cd9856e2e06" />

## DESIGN STEPS

## STEP 1:
Import necessary libraries and load the dataset.

## STEP 2:
Encode categorical variables and normalize numerical features.

## STEP 3:
Split the dataset into training and testing subsets.

## STEP 4:
Design a multi-layer neural network with appropriate activation functions.

## STEP 5:
Train the model using an optimizer and loss function.

## STEP 6:
Evaluate the model and generate a confusion matrix.

## STEP 7:
Use the trained model to classify new data samples.

## STEP 8:
Display the confusion matrix, classification report, and predictions.

### Name:DAKSHA C
### Register Number:212224230048

```
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

  # Initialize model
model = PeopleClassifier(input_size = X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=100)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

  if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


```


## Dataset Information
<img width="1055" height="552" alt="image" src="https://github.com/user-attachments/assets/ece1cab7-3910-41ab-9cfe-caa4ae593eaa" />


## OUTPUT
### Confusion Matrix

<img width="714" height="674" alt="image" src="https://github.com/user-attachments/assets/db1147db-9ae7-4ad7-871b-7236628baea4" />

### Classification Report

<img width="644" height="443" alt="image" src="https://github.com/user-attachments/assets/8ea149a3-3a29-4bf6-9221-f0003e3b8138" />

### New Sample Data Prediction
<img width="428" height="95" alt="image" src="https://github.com/user-attachments/assets/c1346746-5320-454d-a757-d7e02aa566d2" />



## RESULT
Thus the neural network classification model was successfully developed.
## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
