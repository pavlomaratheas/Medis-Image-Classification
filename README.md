In this project we developed a classification model based on DenseNet121 (implemented through MONAI), adapted to three classes corresponding to heart, liver, and lung. Images are preprocessed by resizing them to 224×224 pixels, converting them into tensors, and normalizing with ImageNet statistics; for the training set we also apply simple data augmentation (random horizontal flip and 15° rotation) to improve generalization. The optimization is performed with the Adam algorithm (learning rate 1e−3) and the CrossEntropyLoss function, while the evaluation relies on accuracy, precision, recall, F1-score, and confusion matrix. Training is organized through 5-Fold Stratified Cross-Validation, providing a more robust estimate of performance. The key hyperparameters are: batch size 16, 15 training epochs, and learning rate 1e−3.

To reproduce the results, please follow this setup:

1. Clone the repository:
   git clone https:[//github.com/username/repo-name.git](https://github.com/pavlomaratheas/Medis-Image-Classification
   cd repo-name
   
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

Performance per class:
Heart → Precision 0.80, Recall 0.80, F1 = 0.80
Liver → Precision 0.97, Recall 0.99, F1 = 0.98
Lung → Precision 0.93, Recall 0.89, F1 = 0.91

Example Outputs:
<img width="4470" height="1466" alt="average_metrics" src="https://github.com/user-attachments/assets/785844b5-03f8-4436-be50-e872f3c87584" />



In addition to the main PyTorch Lightning implementation, the repository also includes the files model.py and train_eval.py. These were developed as an alternative exercise, first by adapting a ResNet-based model and then by implementing a simple Convolutional Neural Network (CNN) without using PyTorch Lightning. Their purpose was mainly educational, to explore different approaches and gain a better understanding of model design and basic evaluation metrics.
