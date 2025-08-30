"""
Base Neural Network Classes for MASA Framework
Implements the foundational neural network components used across all agents.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod


class BaseNeuralLayer(nn.Module, ABC):
    """Base class for all neural network layers in the MASA framework."""
    
    def __init__(self, input_size: int, output_size: int, batch_size: int = 32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation."""
        pass
    
    def get_type(self) -> str:
        """Return the type identifier for this layer."""
        return self.__class__.__name__


class TransposeLayer(BaseNeuralLayer):
    """Transpose layer for reshaping tensor dimensions."""
    
    def __init__(self, dim1: int, dim2: int, batch_size: int = 32):
        super().__init__(dim1 * dim2, dim2 * dim1, batch_size)
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose the input tensor."""
        # Reshape and transpose: (batch, dim1*dim2) -> (batch, dim1, dim2) -> (batch, dim2, dim1)
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.dim1, self.dim2)
        return x_reshaped.transpose(-2, -1).contiguous().view(batch_size, -1)


class PiecewiseLinearRepresentation(BaseNeuralLayer):
    """Piecewise Linear Representation layer for time series analysis."""
    
    def __init__(self, units_count: int, window: int, batch_size: int = 32):
        super().__init__(units_count * window, units_count * window, batch_size)
        self.units_count = units_count
        self.window = window
        
        # Learnable parameters for PLR
        self.weight = nn.Parameter(torch.randn(units_count, window, window))
        self.bias = nn.Parameter(torch.zeros(units_count, window))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply piecewise linear representation."""
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.units_count, self.window)
        
        # Apply learnable linear transformation
        output = torch.bmm(x_reshaped, self.weight.expand(batch_size, -1, -1, -1).view(-1, self.window, self.window))
        output = output.view(batch_size, self.units_count, self.window)
        output = output + self.bias.unsqueeze(0)
        
        return output.view(batch_size, -1)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for attention mechanisms."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create relative position embeddings
        self.relative_positions = nn.Parameter(
            torch.randn(2 * max_len - 1, d_model)
        )
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate relative positional encodings."""
        positions = torch.arange(seq_len, device=self.relative_positions.device)
        relative_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_pos = relative_pos + self.max_len - 1
        relative_pos = torch.clamp(relative_pos, 0, 2 * self.max_len - 2)
        
        return self.relative_positions[relative_pos]


class RelativeSelfAttention(BaseNeuralLayer):
    """Self-attention with relative positional encoding."""
    
    def __init__(self, d_model: int, n_heads: int, seq_len: int, batch_size: int = 32):
        super().__init__(d_model * seq_len, d_model * seq_len, batch_size)
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.relative_pos_encoding = RelativePositionalEncoding(self.head_dim, seq_len)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with relative positional attention."""
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.seq_len, self.d_model)
        
        # Linear transformations
        Q = self.q_linear(x_reshaped)
        K = self.k_linear(x_reshaped)
        V = self.v_linear(x_reshaped)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention with relative positions
        attention_output = self._compute_attention_with_relative_pos(Q, K, V)
        
        # Reshape and apply output linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, self.seq_len, self.d_model
        )
        output = self.out_linear(attention_output)
        
        return output.view(batch_size, -1)
    
    def _compute_attention_with_relative_pos(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute attention scores with relative positional encoding."""
        batch_size, n_heads, seq_len, head_dim = Q.shape
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # Add relative positional encoding
        rel_pos_enc = self.relative_pos_encoding(seq_len)
        rel_pos_scores = torch.einsum('bhid,ijd->bhij', Q, rel_pos_enc)
        scores = scores + rel_pos_scores
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output


class RelativeCrossAttention(BaseNeuralLayer):
    """Cross-attention with relative positional encoding for dual input streams."""
    
    def __init__(self, d_model: int, n_heads: int, seq_len_q: int, seq_len_kv: int, batch_size: int = 32):
        super().__init__(d_model * seq_len_q, d_model * seq_len_q, batch_size)
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.relative_pos_encoding = RelativePositionalEncoding(self.head_dim, max(seq_len_q, seq_len_kv))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Forward pass with cross-attention between two input streams."""
        batch_size = query.shape[0]
        
        # Reshape inputs
        query_reshaped = query.view(batch_size, self.seq_len_q, self.d_model)
        kv_reshaped = key_value.view(batch_size, self.seq_len_kv, self.d_model)
        
        # Linear transformations
        Q = self.q_linear(query_reshaped)
        K = self.k_linear(kv_reshaped)
        V = self.v_linear(kv_reshaped)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross-attention
        attention_output = self._compute_cross_attention_with_relative_pos(Q, K, V)
        
        # Reshape and apply output linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, self.seq_len_q, self.d_model
        )
        output = self.out_linear(attention_output)
        
        return output.view(batch_size, -1)
    
    def _compute_cross_attention_with_relative_pos(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute cross-attention scores with relative positional encoding."""
        batch_size, n_heads, seq_len_q, head_dim = Q.shape
        seq_len_kv = K.shape[2]
        
        # Standard cross-attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # Add relative positional encoding (simplified for cross-attention)
        rel_pos_enc = self.relative_pos_encoding(max(seq_len_q, seq_len_kv))
        if seq_len_q <= seq_len_kv:
            rel_pos_scores = torch.einsum('bhid,ijd->bhij', Q, rel_pos_enc[:seq_len_q, :seq_len_kv])
        else:
            rel_pos_scores = torch.einsum('bhid,ijd->bhij', Q, rel_pos_enc[:seq_len_q, :seq_len_kv])
        
        scores = scores + rel_pos_scores
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output


class ResidualConvBlock(BaseNeuralLayer):
    """Residual convolutional block for feature extraction and forecasting."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, batch_size: int = 32):
        super().__init__(in_channels, out_channels, batch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        
        # Projection layer for residual connection if dimensions don't match
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, 1)
            
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        batch_size = x.shape[0]
        seq_len = x.shape[1] // self.in_channels
        
        # Reshape for conv1d: (batch, channels, sequence)
        x_reshaped = x.view(batch_size, self.in_channels, seq_len)
        
        # First convolution
        out = self.conv1(x_reshaped)
        out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.batch_norm2(out)
        
        # Residual connection
        if self.projection is not None:
            x_reshaped = self.projection(x_reshaped)
        out = out + x_reshaped
        
        out = self.activation(out)
        
        # Reshape back to original format
        return out.view(batch_size, -1)


class SAMOptimizedLinear(BaseNeuralLayer):
    """Linear layer with SAM (Sharpness-Aware Minimization) optimization support."""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'sigmoid', batch_size: int = 32):
        super().__init__(input_size, output_size, batch_size)
        
        self.linear = nn.Linear(input_size, output_size)
        self.activation_name = activation
        
        # Initialize activation function
        if activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with specified activation."""
        out = self.linear(x)
        return self.activation(out)


class PSformerBlock(BaseNeuralLayer):
    """PSformer block for sequential pattern analysis."""
    
    def __init__(self, d_model: int, units_count: int, segments: int, rho: float = 0.5, batch_size: int = 32):
        super().__init__(d_model * units_count, d_model * units_count, batch_size)
        self.d_model = d_model
        self.units_count = units_count
        self.segments = segments
        self.rho = rho
        
        # Segmentation parameters
        self.segment_embedding = nn.Linear(d_model, d_model)
        self.position_embedding = nn.Parameter(torch.randn(segments, d_model))
        
        # Attention mechanism
        self.self_attention = RelativeSelfAttention(d_model, 8, segments, batch_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PSformer block."""
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.units_count, self.d_model)
        
        # Segment the input
        segmented = self._segment_input(x_reshaped)
        
        # Add positional embeddings
        segmented = segmented + self.position_embedding.unsqueeze(0)
        
        # Self-attention with residual connection
        attn_input = segmented.view(batch_size, -1)
        attn_output = self.self_attention(attn_input)
        attn_output = attn_output.view(batch_size, self.segments, self.d_model)
        
        segmented = self.norm1(segmented + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(segmented)
        segmented = self.norm2(segmented + self.dropout(ffn_output))
        
        # Reconstruct original shape
        output = self._reconstruct_from_segments(segmented)
        return output.view(batch_size, -1)
    
    def _segment_input(self, x: torch.Tensor) -> torch.Tensor:
        """Segment input into patches with overlap controlled by rho."""
        batch_size, seq_len, d_model = x.shape
        segment_size = seq_len // self.segments
        overlap = int(segment_size * self.rho)
        
        segments = []
        for i in range(self.segments):
            start = max(0, i * segment_size - overlap)
            end = min(seq_len, (i + 1) * segment_size + overlap)
            segment = x[:, start:end, :].mean(dim=1, keepdim=True)
            segments.append(segment)
            
        return torch.cat(segments, dim=1)
    
    def _reconstruct_from_segments(self, segmented: torch.Tensor) -> torch.Tensor:
        """Reconstruct original sequence from segments."""
        batch_size, n_segments, d_model = segmented.shape
        
        # Simple reconstruction by repeating segments
        reconstructed = segmented.repeat_interleave(self.units_count // n_segments, dim=1)
        
        # Handle remainder
        remainder = self.units_count % n_segments
        if remainder > 0:
            extra = segmented[:, :remainder, :]
            reconstructed = torch.cat([reconstructed, extra], dim=1)
            
        return reconstructed


class MASAOptimizer:
    """Custom optimizer implementing SAM (Sharpness-Aware Minimization) for MASA framework."""
    
    def __init__(self, params, base_optimizer=torch.optim.Adam, rho=0.05, **kwargs):
        self.param_groups = list(params)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.rho = rho
        
    def first_step(self, zero_grad=False):
        """First step of SAM optimization - compute adversarial weights."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"
                
        if zero_grad:
            self.zero_grad()
            
    def second_step(self, zero_grad=False):
        """Second step of SAM optimization - update with base optimizer."""
        for group in self.param_groups:
            for p in group:
                if p.grad is None:
                    continue
                p.sub_(p.grad * self.rho / (self._grad_norm() + 1e-12))  # Go back to "w" from "w + e(w)"
                
        self.base_optimizer.step()  # Do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
            
    def step(self, closure=None):
        """Combined step for convenience."""
        assert closure is not None, "SAM requires closure, but it was not provided"
        
        # First forward-backward pass
        closure()
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass
        closure()
        self.second_step()
        
    def zero_grad(self):
        """Zero gradients."""
        self.base_optimizer.zero_grad()
        
    def _grad_norm(self):
        """Compute gradient norm."""
        shared_device = self.param_groups[0][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(dtype=torch.float32).to(shared_device)
                for group in self.param_groups for p in group
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm


# Utility functions for MASA framework
def initialize_weights(module):
    """Initialize weights using Xavier/Glorot initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def entropy_regularization(actions: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute entropy regularization term for action diversity."""
    # Ensure actions are probabilities
    probs = F.softmax(actions / temperature, dim=-1)
    log_probs = F.log_softmax(actions / temperature, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy