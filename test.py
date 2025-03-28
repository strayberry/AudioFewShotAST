from data_loader import FewShotSpeechCommands
from model import ASTEncoder, ProtoNet
from utils import get_accuracy
import torch
import torch.nn.functional as F

def test(model_path):
    # Construct the dataset using the testing subset; target_length must match config (128)
    test_dataset = FewShotSpeechCommands(subset="testing", target_length=128)
    
    # Initialize the model (ProtoNet with embedded ASTEncoder) using the same N-way K-shot configuration
    model = ProtoNet(ASTEncoder(), n_way=5, k_shot=5).cuda()
    
    # Load the saved model weights
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    model.eval()
    with torch.no_grad():
        # Sample an episode, e.g., 5-way 5-shot 5-query
        support, query, labels = test_dataset.sample_episode(n_way=5, k_shot=5, q_query=5)
        support, query, labels = support.cuda(), query.cuda(), labels.cuda()
        
        # Forward pass to obtain the distance matrix (shape [n_way*q_query, n_way], e.g., [25, 5])
        dists = model(support, query)
        
        # Predict classes: smaller distance corresponds to higher probability, so use negative distances
        preds = torch.argmax(-dists, dim=1)
        acc = get_accuracy(dists, labels)
        
        print("Test Accuracy: {:.2f}%".format(acc))
        print("Predictions:", preds.cpu().numpy())
        print("Ground Truth:", labels.cpu().numpy())


if __name__ == "__main__":
    model_path = "./data/model_20250329_211830.pt"
    test(model_path)
