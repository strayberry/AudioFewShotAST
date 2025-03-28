import os
import torch
import torch.nn.functional as F
import logging
import datetime
from data_loader import FewShotSpeechCommands
from model import ASTEncoder, ProtoNet
from utils import get_accuracy

def train():
    # Create log directory and configure logging
    os.makedirs("./log", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("./log", f"train_{timestamp}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    # Initialize training and validation datasets.
    train_dataset = FewShotSpeechCommands(subset="training", target_length=128)
    val_dataset = FewShotSpeechCommands(subset="validation", target_length=128)
    
    # Initialize the model (ProtoNet with ASTEncoder) and move it to GPU
    model = ProtoNet(ASTEncoder(), n_way=5, k_shot=5).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_episodes = 1000
    validation_interval = 100

    model.train()
    for episode in range(num_episodes):
        # Sample an episode: 5-way, 5-shot, 5-query; total 25 samples
        support, query, labels = train_dataset.sample_episode(n_way=5, k_shot=5, q_query=5)
        support, query, labels = support.cuda(), query.cuda(), labels.cuda()

        # Forward pass: ProtoNet.forward returns the distance matrix with shape [25, 5]
        dists = model(support, query)
        
        # Compute loss using cross entropy (using negative distances as logits)
        loss = F.cross_entropy(-dists, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log training progress every 10 episodes
        if episode % 10 == 0:
            train_acc = get_accuracy(dists, labels)
            msg = f"[Episode {episode}] Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f}%"
            print(msg)
            logging.info(msg)
        
        # Validate every 'validation_interval' episodes
        if episode % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_support, val_query, val_labels = val_dataset.sample_episode(n_way=5, k_shot=5, q_query=5)
                val_support, val_query, val_labels = val_support.cuda(), val_query.cuda(), val_labels.cuda()
                val_dists = model(val_support, val_query)
                val_loss = F.cross_entropy(-val_dists, val_labels)
                val_acc = get_accuracy(val_dists, val_labels)
                val_msg = f"[Episode {episode}] Validation Loss: {val_loss.item():.4f} | Validation Acc: {val_acc:.2f}%"
                print(val_msg)
                logging.info(val_msg)
            model.train()
    
    # Save final model with timestamp
    os.makedirs("./data", exist_ok=True)
    model_path = os.path.join("./data", f"model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    logging


if __name__ == "__main__":
    train()
