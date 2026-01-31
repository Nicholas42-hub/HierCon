import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq


class SSLModel(nn.Module):
    """
    Self-Supervised Learning model wrapper for XLS-R
    Extracts features from pre-trained XLS-R model with trainable parameters
    """
    def __init__(self, device, cp_path='/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt'):
        super(SSLModel, self).__init__()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        
        # Keep SSL model parameters trainable (no freezing)
        # All parameters will be fine-tuned during training
        self.model.train()

    def extract_feat(self, input_data):
        """
        Extract features from XLS-R model
        Args:
            input_data: Input audio tensor (B, T) or (B, T, C)
        Returns:
            x: Final layer output (B, T, C)
            layer_results: All layer outputs for layer-wise fusion
        """
        # Ensure model is on correct device and dtype
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
        
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
            
        result = self.model(input_tmp, mask=False, features_only=True)
        return result['x'], result['layer_results']


def getAttenF(layerResult):
    """
    Extract and pool layer features from XLS-R layer results
    Args:
        layerResult: List of layer outputs from XLS-R
    Returns:
        layery: Pooled layer representations (B, L, C)
        fullfeature: Full temporal features (B, L, T, C)
    """
    poollayerResult = []
    fullf = []
    
    for layer in layerResult:
        x = layer[0].transpose(0, 1)  # (B, T, C)
        
        # Pooled representation via adaptive average pooling
        layery = F.adaptive_avg_pool1d(x.transpose(1, 2), 1).transpose(1, 2)
        poollayerResult.append(layery)
        
        # Full temporal feature
        fullf.append(x.unsqueeze(1))
    
    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


class AttnPool(nn.Module):
    """
    Attention-based pooling module
    Learns to weight and aggregate temporal/layer features
    """
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: Input features (B, T, C)
            mask: Optional attention mask
        Returns:
            out: Weighted pooled features (B, C)
            a: Attention weights (B, T)
        """
        x = self.layer_norm(x)
        e = torch.tanh(self.proj(x))
        s = self.score(e).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask.bool(), float('-inf'))
        a = torch.softmax(s, dim=1)
        out = torch.sum(a.unsqueeze(-1) * x, dim=1)
        return out, a


class ResidualRefine(nn.Module):
    """
    Residual refinement module with feed-forward network
    Adds non-linear transformation while maintaining residual connection
    """
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return x + self.net(x)


class MarginContrastiveLoss(nn.Module):
    """
    Margin-based Contrastive Loss for audio deepfake detection
    
    Enforces:
    - Positive pairs (same class): minimize distance
    - Negative pairs (different class): enforce margin-based separation
    
    Reference: Adapted from classic contrastive loss (Hadsell et al., 2006)
    for audio anti-spoofing tasks
    """
    def __init__(self, margin=1.0, distance_type='euclidean'):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
        
    def forward(self, features, labels):
        """
        Args:
            features: Projected embeddings (B, D)
            labels: Binary labels (0=bonafide, 1=spoof)
        Returns:
            loss: Margin-based contrastive loss
        """
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # L2 normalize features for stability
        features = F.normalize(features, dim=1, eps=1e-8)
        
        # Compute pairwise distances
        if self.distance_type == 'euclidean':
            distances = torch.cdist(features, features, p=2)
        else:  # cosine distance
            similarity = torch.matmul(features, features.T)
            distances = 1.0 - similarity
        
        # Create label masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float().to(features.device)
        negative_mask = (labels != labels.T).float().to(features.device)
        
        # Remove diagonal (self-similarity)
        eye_mask = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * (1 - eye_mask)
        negative_mask = negative_mask * (1 - eye_mask)
        
        # Check if valid pairs exist
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Positive loss: minimize distance between same class samples
        pos_loss = (distances * positive_mask).pow(2)
        pos_loss = pos_loss.sum() / (positive_mask.sum() + 1e-8)
        
        # Negative loss: enforce margin - only penalize if distance < margin
        margin_violations = F.relu(self.margin - distances)
        neg_loss = (margin_violations * negative_mask).pow(2)
        neg_loss = neg_loss.sum() / (negative_mask.sum() + 1e-8)
        
        total_loss = pos_loss + neg_loss
        
        # Clamp for numerical stability
        return torch.clamp(total_loss, 0, 10.0)


class CompatibleCombinedLoss(nn.Module):
    """
    Combined loss function for audio deepfake detection
    
    Combines:
    - Cross-entropy loss for classification
    - Margin-based contrastive loss for embedding separation
    """
    def __init__(self, weight=None, margin=0.5, contrastive_weight=0.1, distance_type='euclidean'):
        super().__init__()
        self.cce_loss = nn.CrossEntropyLoss(weight=weight)
        self.contrastive_loss = MarginContrastiveLoss(margin=margin, distance_type=distance_type)
        self.contrastive_weight = contrastive_weight
        
    def forward(self, logits, features, labels):
        """
        Args:
            logits: Classification logits (B, num_classes)
            features: Projected features for contrastive learning (B, D)
            labels: Ground truth labels (B,)
        Returns:
            total_loss: Combined weighted loss
            cce: Cross-entropy loss component
            combined_contrastive: Weighted contrastive loss component
        """
        cce = self.cce_loss(logits, labels)
        contrastive = self.contrastive_loss(features, labels)
        combined_contrastive = self.contrastive_weight * contrastive
        total_loss = cce + combined_contrastive
        return total_loss, cce, combined_contrastive


class ModelHierarchicalContrastive(nn.Module):
    """
    Hierarchical Attention Model with Contrastive Learning
    
    Architecture:
    1. XLS-R SSL feature extraction (trainable)
    2. Temporal attention within each layer
    3. Intra-group attention (local layer interactions)
    4. Inter-group attention (global aggregation)
    5. Classification + contrastive learning
    
    Features:
    - Multi-level attention mechanism
    - Interpretability through attention weights
    - Margin-based contrastive learning
    - Full end-to-end fine-tuning
    """
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(device)
        self.d_model = 1024
        self.group_size = getattr(args, "group_size", 3)
        
        # Hierarchical attention modules
        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.15)
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.15)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128)
        )
        
        # Classifier for bonafide vs spoof
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 2),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Storage for attention weights (for interpretability)
        self.attention_weights = {}

    def forward(self, x, return_features=False, return_interpretability=False):
        """
        Forward pass with optional feature/interpretability extraction
        
        Args:
            x: Input audio tensor (B, T) or (B, T, 1)
            return_features: Whether to return projected features for contrastive loss
            return_interpretability: Whether to return attention weights and explanations
            
        Returns:
            output: Log-softmax predictions (B, 2)
            projected_features: (Optional) Features for contrastive learning
            interpretations: (Optional) Dictionary with attention weights and explanations
        """
        # Extract SSL features from XLS-R (with gradient)
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)

        B, L, T, C = fullfeature.shape
        
        # ========== 1. Temporal Attention ==========
        # Aggregate temporal information within each layer
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, temporal_attn = self.temporal_attn(layer_tokens)
        layer_emb = layer_emb.view(B, L, C)
        
        # Store temporal attention for interpretability
        temporal_attn = temporal_attn.view(B, L, T)
        if return_interpretability:
            self.attention_weights['temporal'] = temporal_attn.detach()
        
        # Handle layer grouping (pad if necessary)
        if layer_emb.size(1) % self.group_size != 0:
            pad_size = self.group_size - (layer_emb.size(1) % self.group_size)
            layer_emb = F.pad(layer_emb, (0, 0, 0, pad_size), mode='constant', value=0)
        
        # ========== 2. Intra-group Attention ==========
        # Local interactions within layer groups
        groups = torch.split(layer_emb, self.group_size, dim=1)
        group_vecs = []
        intra_attn_weights = []
        
        for g in groups:
            g_vec, intra_attn = self.intra_attn(g)
            g_vec = self.group_refine(g_vec)
            group_vecs.append(g_vec)
            intra_attn_weights.append(intra_attn)
        
        if return_interpretability:
            self.attention_weights['intra_group'] = torch.stack(
                [a.detach() for a in intra_attn_weights], dim=1
            )
        
        # ========== 3. Inter-group Attention ==========
        # Global aggregation across groups
        group_stack = torch.stack(group_vecs, dim=1)
        utt_emb, inter_attn = self.inter_attn(group_stack)
        utt_emb = self.utt_refine(utt_emb)
        
        if return_interpretability:
            self.attention_weights['inter_group'] = inter_attn.detach()
        
        # ========== 4. Classification ==========
        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        
        # ========== Return Values ==========
        if return_interpretability:
            interpretations = self._compute_interpretations(
                temporal_attn, intra_attn_weights, inter_attn, B, L, T
            )
            if return_features:
                projected_features = self.projection_head(utt_emb)
                return output, projected_features, interpretations
            return output, interpretations
        
        if return_features:
            projected_features = self.projection_head(utt_emb)
            return output, projected_features
        
        return output
    
    def _compute_interpretations(self, temporal_attn, intra_attn_list, inter_attn, B, L, T):
        """
        Compute interpretability metrics from attention weights
        
        Args:
            temporal_attn: Temporal attention weights (B, L, T)
            intra_attn_list: List of intra-group attention weights
            inter_attn: Inter-group attention weights (B, num_groups)
            B, L, T: Batch size, number of layers, time steps
            
        Returns:
            Dictionary containing:
            - layer_importance: Importance score for each layer
            - temporal_importance: Importance score for each time step
            - attention_weights: Raw attention weights
            - text_explanations: Human-readable explanations
        """
        # 1. Compute layer importance
        layer_importance = self._compute_layer_importance(intra_attn_list, inter_attn, L)
        
        # 2. Compute temporal importance
        temporal_importance = self._compute_temporal_importance(temporal_attn, layer_importance)
        
        # 3. Generate text explanations
        explanations = self._generate_explanations(layer_importance, temporal_importance)
        
        return {
            'layer_importance': layer_importance,  # (B, L)
            'temporal_importance': temporal_importance,  # (B, T)
            'attention_weights': self.attention_weights,
            'text_explanations': explanations
        }
    
    def _compute_layer_importance(self, intra_attn_list, inter_attn, L):
        """
        Compute importance score for each XLS-R layer
        Combines inter-group and intra-group attention weights
        """
        B = inter_attn.size(0)
        num_groups = len(intra_attn_list)
        
        layer_importance = []
        for group_idx in range(num_groups):
            group_weight = inter_attn[:, group_idx:group_idx+1]  # (B, 1)
            intra_weight = intra_attn_list[group_idx]  # (B, group_size)
            layer_weight = group_weight * intra_weight  # (B, group_size)
            layer_importance.append(layer_weight)
        
        layer_importance = torch.cat(layer_importance, dim=1)[:, :L]
        
        # Normalize to sum to 1
        layer_importance = layer_importance / (layer_importance.sum(dim=1, keepdim=True) + 1e-8)
        return layer_importance
    
    def _compute_temporal_importance(self, temporal_attn, layer_importance):
        """
        Compute importance score for each time step
        Weighted combination of temporal attention across layers
        """
        B, L, T = temporal_attn.shape
        layer_importance_expanded = layer_importance.unsqueeze(2)  # (B, L, 1)
        weighted_temporal_attn = temporal_attn * layer_importance_expanded  # (B, L, T)
        temporal_importance = weighted_temporal_attn.sum(dim=1)  # (B, T)
        
        # Normalize to sum to 1
        temporal_importance = temporal_importance / (temporal_importance.sum(dim=1, keepdim=True) + 1e-8)
        return temporal_importance
    
    def _generate_explanations(self, layer_importance, temporal_importance):
        """
        Generate human-readable explanations for model decisions
        
        Analyzes:
        - Which XLS-R layers contributed most to the decision
        - Which temporal regions (early/middle/late) were most important
        """
        B = layer_importance.size(0)
        explanations = []
        
        for i in range(B):
            layer_imp = layer_importance[i]
            temporal_imp = temporal_importance[i]
            
            # Find top 3 most important layers
            top_layers = torch.topk(layer_imp, k=min(3, len(layer_imp)))
            top_layer_indices = top_layers.indices.cpu().numpy()
            top_layer_values = top_layers.values.cpu().numpy()
            
            # Analyze temporal regions (early/middle/late)
            T = temporal_imp.size(0)
            early = temporal_imp[:T//3].sum().item()
            middle = temporal_imp[T//3:2*T//3].sum().item()
            late = temporal_imp[2*T//3:].sum().item()
            
            # Generate explanation text
            explanation = "Detection decision based on:\n"
            explanation += f"- Key layers: {top_layer_indices[0]}({top_layer_values[0]:.1%}), "
            explanation += f"{top_layer_indices[1]}({top_layer_values[1]:.1%}), "
            explanation += f"{top_layer_indices[2]}({top_layer_values[2]:.1%})\n"
            
            temporal_regions = [('early', early), ('middle', middle), ('late', late)]
            temporal_regions.sort(key=lambda x: x[1], reverse=True)
            explanation += f"- Temporal focus: {temporal_regions[0][0]}({temporal_regions[0][1]:.1%}), "
            explanation += f"{temporal_regions[1][0]}({temporal_regions[1][1]:.1%})"
            
            explanations.append(explanation)
        
        return explanations


# Maintain backward compatibility
CombinedLoss = CompatibleCombinedLoss
Model = ModelHierarchicalContrastive