import time
import numpy as np
num_classes = len(df['topic'].unique())
multiclass_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device)
multiclass_recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
multiclass_precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
multiclass_f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)

def train(model, dataloader, optimizer, criterion) -> Tuple[float, float, float, float, float]:
    """
    Trains the model on the provided training dataset.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function.
    Returns:
        tuple: accuracy, recall, precision, f1_score, and loss of the model on the training dataset.
    """
    # Change model to training mode
    model.train()
    
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    losses = []
    
    for idx, (texts, labels) in enumerate(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        texts = texts.to(device)
        # Start time:
        start = time.perf_counter()
        
        # Reset gradients:
        optimizer.zero_grad()

        # Produce model output:
        predicted_labels = model(texts)
        
        # Calculate loss:
        loss = criterion(predicted_labels, labels)
        losses.append(loss.item())

        # Back Propagation:
        loss.backward()
        
        # Update model parameters:
        optimizer.step()

        # Evaluate performance between predicted labels and true labels:
        accuracies.append(multiclass_accuracy(predicted_labels.argmax(1), labels).item())
        recalls.append(multiclass_recall(predicted_labels.argmax(1), labels).item())
        precisions.append(multiclass_precision(predicted_labels.argmax(1), labels).item())
        f1_scores.append(multiclass_f1_score(predicted_labels.argmax(1), labels).item())

        # Calculate elapsed time:
        elapsed_time = int((time.perf_counter() - start) * 1000)

        # Print:
        print(
            "Training: {}/{} - {} ms/step - accuracy {:.4f} - recall {:.4f}" 
            " - precision {:.4f} - f1 score {:.4f} - loss {:.4f}"\
            .format(idx+1, len(dataloader), elapsed_time, np.mean(accuracies), np.mean(recalls), np.mean(precisions), np.mean(f1_scores), np.mean(losses)), end='\r')
    
    return (
        np.mean(accuracies), 
        np.mean(recalls), 
        np.mean(precisions), 
        np.mean(f1_scores), 
        np.mean(losses) 
    )