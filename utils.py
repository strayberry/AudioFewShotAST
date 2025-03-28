

def get_accuracy(dists, labels):
    # Compute the accuracy of predictions based on distances
    preds = (-dists).argmax(dim=1)
    return (preds == labels).float().mean().item() * 100
