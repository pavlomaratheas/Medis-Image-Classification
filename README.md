In this project we developed a classification model based on DenseNet121 (implemented through MONAI), adapted to three classes corresponding to heart, liver, and lung. Images are preprocessed by resizing them to 224×224 pixels, converting them into tensors, and normalizing with ImageNet statistics; for the training set we also apply simple data augmentation (random horizontal flip and 15° rotation) to improve generalization. The optimization is performed with the Adam algorithm (learning rate 1e−3) and the CrossEntropyLoss function, while the evaluation relies on accuracy, precision, recall, F1-score, and confusion matrix. Training is organized through 5-Fold Stratified Cross-Validation, providing a more robust estimate of performance. The key hyperparameters are: batch size 16, 15 training epochs, and learning rate 1e−3.

To reproduce the results, please follow this setup:

1. Clone the repository:
   git clone https://github.com/pavlomaratheas/Medis-Image-Classification.git
   cd Medis-Image-Classification
   
2. Create and activate a virtual environment (Python 3.10+):
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Run the training with:
   python train_lightning.py

This will: extract the dataset from Data.zip, prepare training/validation folds, train the model with 5-Fold Cross-Validation, print classification reports and confusion matrices, save logs and plots: Training & Validation Curves (loss, accuracy), Confusion Matrix, Classification Report, Per-Class Metrics. All results are stored in the plots/ folder.

Results:
Final results with 15 epochs of training across 5 folds:
Overall Accuracy: 94.4%
Macro Precision: 0.90
Macro Recall: 0.89
Macro F1: 0.90

Example Outputs:
Training and Validation Curves
<img width="4470" height="1466" alt="average_metrics" src="https://github.com/user-attachments/assets/785844b5-03f8-4436-be50-e872f3c87584" />
The following plots show the average loss and average accuracy across the 5 folds, including the standard deviation (shaded areas).

- Loss curves (left):  
  The training loss (blue) is close to zero from the beginning, while the validation loss (red) starts high and decreases rapidly, stabilizing near zero.  
  The shaded red area shows variability across folds, which is initially large but shrinks over time.  
  This indicates that the model quickly learns to minimize the loss and generalizes well without strong signs of overfitting.

- Accuracy curves (right):  
  The training accuracy (green) stays at 100%, showing that the model can perfectly fit the training data.  
  The validation accuracy (orange) increases steadily from ~0.37 to ~0.9, with reduced variability across folds in later epochs.  
  This matches the final classification report (~94% accuracy) and shows that the model generalizes well, even though class imbalance slightly affects performance.

Per class metrics
<img width="3570" height="1770" alt="per_class_metrics" src="https://github.com/user-attachments/assets/0246fc32-43c3-4ac8-95a6-7b8d6bbaaba2" />
This bar chart shows precision, recall, and F1-score for each class:

- Heart: All three metrics are around **0.80**, confirming that this class is the most challenging due to the small number of samples.  
- Liver: Metrics are close to **1.0**, indicating excellent performance and strong robustness thanks to the large sample size.  
- Lung: Precision, recall, and F1 are around **0.90–0.93**, showing reliable classification with only a few misclassifications.  

These results highlight the impact of **class imbalance**: the class with fewer samples (Heart) has lower metrics, while the majority class (Liver) achieves the highest performance.

Confusion_matrix
<img width="2826" height="2370" alt="confusion_matrix" src="https://github.com/user-attachments/assets/17a0e5cb-697a-4afd-8974-3948d9ca98e0" />
The confusion matrix shows how predictions are distributed across the three classes:

- Heart: Out of 20 real Heart samples, 16 were correctly classified, 4 were misclassified as Lung.  
- Liver: Out of 131 real Liver samples, 130 were correctly classified, with only 1 misclassified as Heart.  
- Lung: Out of 63 real Lung samples, 56 were correctly classified, while 7 were misclassified (3 as Heart, 4 as Liver).  

The model achieves near-perfect classification for Liver, while Heart is more challenging due to fewer training samples. Most errors happen between Heart and Lung, which might indicate similarity in certain features of these classes.

Classification Report:
<img width="2385" height="1112" alt="classification_report" src="https://github.com/user-attachments/assets/196088ae-22ff-40e7-87ed-ac78fd762082" />
The table below summarizes the model’s performance across the three classes.

- Heart: Precision = 0.80, Recall = 0.80, F1 = 0.80.  
  Performance is lower compared to other classes due to the limited number of samples (support = 20).  

- Liver: Precision = 0.97, Recall = 0.99, F1 = 0.98.  
  Excellent results, with the model achieving very high recall thanks to the large number of samples (support = 131).  

- Lung: Precision = 0.93, Recall = 0.89, F1 = 0.91.  
  Solid performance, although slightly lower recall suggests some misclassifications.  

The support column indicates how many samples belong to each class in the evaluation set, highlighting the dataset imbalance.  
Overall accuracy is 94.4%, with macro-averaged precision/recall/F1 around 0.90.

In addition to the main PyTorch Lightning implementation, the repository also includes the files model.py and train_eval.py. These were developed as an alternative exercise, first by adapting a ResNet-based model and then by implementing a simple Convolutional Neural Network (CNN) without using PyTorch Lightning. Their purpose was mainly educational, to explore different approaches and gain a better understanding of model design and basic evaluation metrics.
