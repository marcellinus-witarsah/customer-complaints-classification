def evaluate(model, dataloader, criterion) -> Tuple[float, float, float, float, float]:
    """Evaluates the model on the provided validation dataset (usually).

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation data.
    Returns:
        tuple: accuracy, recall, precision, and f1_score of the model on the evaluation dataset.
    """
    # Change model to evaluation mode
    model.eval()
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    losses = []
    
    with torch.no_grad():
        for idx, (texts, labels) in enumerate(dataloader):
            texts, labels = texts.to(device), labels.to(device)
            
            # Start time:
            start = time.perf_counter()
            
            # Produce model output:
            predicted_labels = model(texts)

            # Calculate loss:
            loss = criterion(predicted_labels, labels)
            losses.append(loss.item())

            # Evaluate performance between predicted labels and true labels:
            accuracies.append(multiclass_accuracy(predicted_labels.argmax(1), labels).item())
            recalls.append(multiclass_recall(predicted_labels.argmax(1), labels).item())
            precisions.append(multiclass_precision(predicted_labels.argmax(1), labels).item())
            f1_scores.append(multiclass_f1_score(predicted_labels.argmax(1), labels).item())

            # Calculate elapsed time:
            elapsed_time = int((time.perf_counter() - start) * 1000)
            
            # Print:
            print(
            "Validation: {}/{} - {} ms/step - accuracy {:.4f} - recall {:.4f}" 
            " - precision {:.4f} - f1 score {:.4f} - loss {:.4f}"\
            .format(idx+1, len(dataloader), elapsed_time, np.mean(accuracies), np.mean(recalls), np.mean(precisions), np.mean(f1_scores), np.mean(losses)), end='\r')
    
    return (
        np.mean(accuracies), 
        np.mean(recalls), 
        np.mean(precisions), 
        np.mean(f1_scores),
        np.mean(losses)
    )