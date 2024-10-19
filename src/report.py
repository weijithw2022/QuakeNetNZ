# This file handles the reporting utilities and other helper functions

import matplotlib.pyplot as plt
from fpdf import FPDF
import os

def plot_loss(train_losses, file_name):
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    image_filename = f"{file_name}.jpg"
    plt.savefig(image_filename)  # Save as PNG, you can change to other formats like .pdf, .jpeg
    print(f"Loss curve saved as {image_filename}")
    
def accuracy():
    return 0

def precision():
    return 0

def recall():
    return 0

def F1Score():
    return 0

def test_report(cfg, model, true_tensor, predicted_classes):

    TP = ((predicted_classes == 1) & (true_tensor == 1)).sum().item()  # True Positives
    TN = ((predicted_classes == 0) & (true_tensor == 0)).sum().item()  # True Negatives
    FP = ((predicted_classes == 1) & (true_tensor == 0)).sum().item()  # False Positives
    FN = ((predicted_classes == 0) & (true_tensor == 1)).sum().item()  # False Negatives

    # Calculate Accuracy, Precision, Recall, and F1 Score
    accuracy = 100*((TP + TN) / (TP + TN + FP + FN))
    precision = 100*(TP / (TP + FP)) if (TP + FP) != 0 else 0
    recall = 100*(TP / (TP + FN)) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Print the results
    print(f'Accuracy: {accuracy:.4f}%')
    print(f'Precision: {precision:.4f}%')
    print(f'Recall: {recall:.4f}%')
    print(f'F1 Score: {f1:.4f}%')
    
    pdf = FPDF()
    pdf.add_page()

    # Add the model ID as the title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt=f"Model: {model.model_id}", ln=True, align='C')

    # Add the loss curve title and image
    pdf.cell(200, 10, txt="Training Loss Curve", ln=True, align='C')

    loss_curve_image = cfg.MODEL_PATH + model.model_id + ".jpg"
    
    if os.path.exists(loss_curve_image):
        pdf.image(loss_curve_image, x=10, y=30, w=180)  # Adjust the position (x, y) and size (w)
        pdf.ln(150)  
    else:
        print(f"Loss curve not found for {model.model_id}")

    pdf.set_font("Arial", size=12)

    # Add the calculated metrics to the PDF
    pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Precision: {precision:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Recall: {recall:.4f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"F1 Score: {f1:.4f}%", ln=True, align='L')

    pdf_filename = cfg.MODEL_PATH + model.model_id + ".pdf"
    pdf.output(pdf_filename)

    return 0

def find_latest_file(directory, prefix, extension):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if not files:
        raise ValueError(f"No files with prefix '{prefix}' and extension '{extension}' found in {directory}")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return latest_file

def getLatestModelName(cfg):
    # Fine the latest file name
    directory = cfg.MODEL_PATH  # Change to your model directory
    model_prefix = (cfg.MODEL_TYPE.name).lower()
    model_extension = ".pt"
    latest_model_file = find_latest_file(directory, model_prefix, model_extension)
    return latest_model_file