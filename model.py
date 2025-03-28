from transformers import AutoConfig, AutoModel
import torch.nn as nn
import torch

class ASTEncoder(nn.Module):
    def __init__(self, pretrained_model_dir="./data/ast_model"):
        super().__init__()
        
        # Model configuration
        self.config = AutoConfig.from_pretrained(pretrained_model_dir)
        self.ast = AutoModel.from_pretrained(
            pretrained_model_dir,
            config=self.config,
            ignore_mismatched_sizes=True
        )
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.hidden_size//2),
            nn.GELU()
        )

    def forward(self, x):
        """Input shape: [batch, 128, 128], Output: [batch, 384]"""
        outputs = self.ast(input_values=x)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.projection(pooled)

class ProtoNet(nn.Module):
    def __init__(self, encoder, n_way, k_shot):
        super().__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.k_shot = k_shot
        
    def compute_distance(self, support_emb, query_emb):
        """Compute Euclidean distance between queries and prototypes"""
        prototypes = support_emb.view(self.n_way, self.k_shot, -1).mean(dim=1)
        return torch.cdist(query_emb, prototypes)

    def forward(self, support, query):
        """Input shapes: [n_way*k_shot, 128, 128] and [n_way*q_query, 128, 128]"""
        support_emb = self.encoder(support)
        query_emb = self.encoder(query)
        return self.compute_distance(support_emb, query_emb)


if __name__ == "__main__":
    # Unit tests
    print("=== Model Tests ===")
    
    # Test ASTEncoder
    encoder = ASTEncoder()
    dummy_input = torch.randn(25, 128, 128)
    output = encoder(dummy_input)
    print(f"Encoder output shape: {output.shape} (Expected: [25, 384])")
    
    # Test ProtoNet
    protonet = ProtoNet(encoder, 5, 5)
    dists = protonet(torch.randn(25, 128, 128), torch.randn(25, 128, 128))
    print(f"Distance matrix shape: {dists.shape} (Expected: [25, 5])")