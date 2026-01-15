import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from plm_special.utils.plm_utils import load_plm_llama
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from baseline_special.utils.constants import BITRATE_LEVELS
import pickle
from munch import Munch
from plm_special.data.dataset import ExperienceDataset
from pprint import pprint

class ABRLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        #Arguments prepare
        self.args = args
        self.llm_dim = args.llm_dim
        self.tiny_vocab_size = 1000
        self.state_embedding_dim = args.state_embedding_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_use_self_attention = args.state_use_self_attention
        self.state_attn_hidden_dim = args.state_attn_hidden_dim
        self.fusion_method = args.fusion_method

        #self.data_description = "Input sequence: [instruction, (return, state, action) for each timestep]. State: last bitrate, buffer size, past throughput, past download time, next chunk sizes, remaining chunks. Action: bitrate level (0-5). Return: cumulative reward."
        #self.task_description = "Predict bitrate level (0-5) for next chunk to maximize QoE."
        self.data_description = "The data is a sequence of network states for adaptive bitrate streaming. Each state consists of six features: last bitrate, current buffer size, past k throughput measurements, past k download times, next chunk sizes for different bitrates, and remaining chunks to download."
        self.task_description = "Based on the given network states, predict the optimal bitrate level for the next video chunk to maximize user Quality of Experience (QoE) by balancing video quality, rebuffering events, and smoothness of playback."
        self.max_length = args.max_length

        #load llm&tokenizer
        self.plm, self.tokenizer, self.plm_config = load_plm_llama(args.model_path)
        
        # Alias for compatibility with training framework
        self.plm_embed_size = self.llm_dim  # For compatibility with training framework
        self.bitrate_levels = BITRATE_LEVELS  # For compatibility with training framework

        #frozen llm
        if args.frozen:
            for param in self.plm.parameters():
                param.requires_grad = False
        self.plm = self.plm.to(self.device)
        # Keep PLM in float32 to avoid dtype mismatch with LoRA layers
        
        #vocabulary prepare
        self.word_embeddings = self.plm.get_input_embeddings().weight  #torch.Size([128257, 2048])
        self.vocabulary_size = self.word_embeddings.shape[0]
        assert(len(self.tokenizer) == self.vocabulary_size)
        #networks4ABRLLM
        # Note: action and return are now processed in Encoder, so we don't need separate embedding layers
        self.timestep_embedding = nn.Embedding(100, self.llm_dim).to(self.device)
        self.mapping_layer = nn.Linear(self.vocabulary_size, self.tiny_vocab_size).to(self.device)
        
        # Now we always use self-attention with unified features (state + action + return = 8 features)
        # So alignment_input_dim is always the hidden_dim from self-attention
        alignment_input_dim = self.state_attn_hidden_dim
        
        self.encoder = Encoder(
            self.state_use_self_attention, 
            embed_dim=self.state_embedding_dim,
            hidden_dim=self.state_attn_hidden_dim,
            fusion_method=self.fusion_method
        ).to(self.device)
        self.alignment_layer = AlignmentLayer(alignment_input_dim, args.num_heads, args.key_dim, self.llm_dim).to(self.device)
        self.action_projection = nn.Linear(self.llm_dim, BITRATE_LEVELS).to(self.device)
        
        # modules_except_plm: used for saving/loading modules except PLM (for compatibility with training framework)
        self.modules_except_plm = nn.ModuleList([
            self.encoder,
            self.timestep_embedding,
            self.mapping_layer,
            self.alignment_layer,
            self.action_projection
        ])
        
        # Store historical unified embeddings (state+action+return are unified now)
        # Use float32 for consistency (will convert to bfloat16 when needed for LLM)
        self.unified_history_dq = deque([torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device)], maxlen=self.max_length)

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        states = states.to(self.device).float()  # shape: (batch_size, seq_len, 6, 6), dtype: float32
        actions = actions.to(self.device).float()  # shape: (batch_size, seq_len, 1), dtype: float32
        returns = returns.to(self.device).float()  # shape: (batch_size, seq_len, 1), dtype: float32
        timesteps = timesteps.to(self.device).long()  # shape: (batch_size, seq_len), dtype: int64 (for Embedding)
        
        # Prepare instruction
        #TODO: fix instruction content
        instruction = (
            f"<|start_prompt|>Data description: {self.data_description}"
            f" Task description: {self.task_description} "
            f" Input statistics: "
            f"<|end_prompt|>"
        )
        instruction = self.tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        instruction_tokens = instruction.to(self.device).long()  # Ensure long dtype for embedding lookup
        instruction_embeddings = self.plm.get_input_embeddings()(instruction_tokens)

        # Tiny vocabulary mapping
        # word_embeddings is float32 (from LLM), mapping_layer also uses float32
        word_embeddings_float = self.word_embeddings.permute(1, 0).to(torch.float32)  # Ensure float32
        tiny_vocab_embeddings = self.mapping_layer(word_embeddings_float).permute(1, 0)  # (vocab_size, llm_dim), dtype: float32

        # Unified encoding: state, action, and return together
        unified_features = self.encoder(states, actions, returns)  # Returns unified hidden representation
        
        # Alignment with tiny vocabulary
        unified_embeddings = self.alignment_layer(unified_features, tiny_vocab_embeddings, tiny_vocab_embeddings)  # (batch, seq_len, llm_dim)
        
        # Add timestep embeddings to unified embeddings
        timestep_embeddings = self.timestep_embedding(timesteps)  # (batch, seq_len, llm_dim)
        unified_embeddings = unified_embeddings + timestep_embeddings

        concated_embeddings = [instruction_embeddings]  # LLM embeddings are float32
        
        #test without prefix prompt
        #concated_embeddings = []

        # Concatenate unified embeddings for each timestep
        concated_embeddings.append(unified_embeddings)  # (batch, seq_len, llm_dim)
        
        concated_embeddings = torch.cat(concated_embeddings, dim=1)  # (batch, total_seq_len, llm_dim), dtype: float32

        # LLM forward
        output = self.plm(inputs_embeds=concated_embeddings).last_hidden_state
        # output shape: (batch, total_seq_len, llm_dim)
        
        # Extract outputs corresponding to unified embeddings positions for action prediction
        # Since unified_embeddings are concatenated right after instruction, we can directly slice
        instruction_len = instruction_embeddings.shape[1]
        unified_outputs = output[:, instruction_len:, :]  # (batch, seq_len, llm_dim), dtype: float32
        
        # Predict actions
        # unified_outputs is already in float32
        action_pred = self.action_projection(unified_outputs)  # (batch, seq_len, BITRATE_LEVELS), dtype: float32
        return action_pred
    
    def sample(self, state, target_return, timestep, **kwargs):
        """
        Sample action function for evaluation/testing.
        Compatible with OfflineRLPolicy interface.
        
        Args:
            state: (1, 1, 6, 6) - current network state
            target_return: float - target return value
            timestep: int - current timestep
        
        Returns:
            bitrate: int - selected bitrate level
        """
        # Step 1: Get historical context from deques
        prev_stacked_inputs = self.get_historical_context()  # (1, hist_len, llm_dim)
        
        # Step 2: Prepare instruction prompt (same as forward)
        #TODO: fix instruction content
        instruction = (
            f"<|start_prompt|>Data description: {self.data_description}"
            f" Task description: {self.task_description} "
            f" Input statistics: "
            f"<|end_prompt|>"
        )
        instruction = self.tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        instruction_tokens = instruction.to(self.device).long()
        instruction_embeddings = self.plm.get_input_embeddings()(instruction_tokens)  # (1, instruction_len, llm_dim)
        
        # Step 3: Tiny vocabulary mapping (same as forward)
        word_embeddings_float = self.word_embeddings.permute(1, 0).to(torch.float32)
        tiny_vocab_embeddings = self.mapping_layer(word_embeddings_float).permute(1, 0)  # (vocab_size, llm_dim)
        
        # Step 4: Process state, action, and return together
        state = state.to(self.device).float()  # (1, 1, 6, 6)
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)  # Placeholder for action in sample
        
        unified_features = self.encoder(state, action_tensor, target_return)  # Returns unified hidden representation
        
        # Alignment with tiny vocabulary
        unified_embeddings = self.alignment_layer(unified_features, tiny_vocab_embeddings, tiny_vocab_embeddings)  # (1, 1, llm_dim)
        
        # Step 5: Add timestep embeddings to unified embeddings
        timestep_tensor = torch.as_tensor(timestep, dtype=torch.long, device=self.device).reshape(1, 1)
        timestep_embeddings = self.timestep_embedding(timestep_tensor)  # (1, 1, llm_dim)
        unified_embeddings = unified_embeddings + timestep_embeddings  # (1, 1, llm_dim)
        
        unified_embeddings_with_timestep = unified_embeddings
        
        # Step 6: Concatenate: [historical, instruction, unified_embeddings]
        # Format matches forward: [instruction, unified_embeddings for each timestep]
        # For sample: [historical(unified_embeddings)..., instruction, unified_embeddings]
        concated_embeddings = []
        
        # Add historical context if exists (already in float32)
        if prev_stacked_inputs.shape[1] > 0:
            concated_embeddings.append(prev_stacked_inputs)
        
        # Add instruction (already in float32)
        concated_embeddings.append(instruction_embeddings)
        
        # Add current unified embedding (already in float32, contains state+action+return)
        concated_embeddings.append(unified_embeddings_with_timestep)
        
        stacked_inputs = torch.cat(concated_embeddings, dim=1)  # (1, total_seq_len, llm_dim)
        
        # Truncate if too long (should not exceed plm max length)
        # Note: plm_embed_size might refer to max sequence length, but we use a reasonable limit
        max_seq_len = 2048  # Reasonable limit for LLM
        if stacked_inputs.shape[1] > max_seq_len:
            stacked_inputs = stacked_inputs[:, -max_seq_len:, :]
        
        # Create attention mask
        attention_mask = torch.ones((stacked_inputs.shape[0], stacked_inputs.shape[1]), dtype=torch.long, device=self.device)
        
        # Step 7: LLM forward
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        output = transformer_outputs['last_hidden_state'] if isinstance(transformer_outputs, dict) else transformer_outputs
        # output shape: (1, total_seq_len, llm_dim)
        
        # Step 8: Predict action from the last unified embedding position
        # The last position corresponds to the unified embedding we just added
        logits_used = output[:, -1:, :]  # (1, 1, llm_dim) - use last position (unified embedding), dtype: float32
        action_pred = self.action_projection(logits_used)  # (1, 1, BITRATE_LEVELS), dtype: float32
        action_pred = action_pred.reshape(-1)  # (BITRATE_LEVELS,)
        
        # Sample action
        bitrate, _ = self._sample(action_pred)
        
        # Step 9: For history update, we use the unified embeddings
        # Since action and return are now part of unified encoding, all three use the same embedding
        state_embeddings = unified_embeddings_with_timestep  # (1, 1, llm_dim)
        action_embeddings = unified_embeddings_with_timestep  # (1, 1, llm_dim)
        return_embeddings = unified_embeddings_with_timestep  # (1, 1, llm_dim)
        
        # Step 10: Update deques
        # All embeddings are already in float32
        self.update_history(state_embeddings, action_embeddings, return_embeddings)
        
        return bitrate
    
    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), weights=pi)[0]
        lgprob = np.log(pi[idx] + 1e-8)  # Add small epsilon to avoid log(0)
        return idx, lgprob
    
    def clear_dq(self):
        """
        Clear the unified history deque and reset it to empty state.
        This should be called at the start of each new episode.
        """
        self.unified_history_dq.clear()
        
        # Reinitialize with empty tensor (use float32 for consistency with embedding outputs)
        self.unified_history_dq.append(torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device))
    
    def get_historical_context(self):
        """
        Get historical context from unified history deque for building sequence input.
        Returns concatenated historical embeddings.
        
        Returns:
            historical_context: (1, total_seq_len, llm_dim) - concatenated historical embeddings
        """
        # Stack historical unified embeddings
        historical_inputs = []
        for i in range(len(self.unified_history_dq)):
            prev_unified = self.unified_history_dq[i]
            # Check if it has non-zero sequence length
            if prev_unified.shape[1] > 0:
                historical_inputs.append(prev_unified)
        
        if len(historical_inputs) > 0:
            return torch.cat(historical_inputs, dim=1)
        else:
            # Return empty tensor if no history (use float32 for consistency)
            return torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device)
    
    def update_history(self, state_embedding, action_embedding, return_embedding):
        """
        Update unified history deque with new unified embedding.
        Since state, action, and return are now unified, we only need to store one embedding.
        
        Args:
            state_embedding: (1, seq_len, llm_dim) - unified embedding (state+action+return)
            action_embedding: (1, seq_len, llm_dim) - same as state_embedding (for compatibility)
            return_embedding: (1, seq_len, llm_dim) - same as state_embedding (for compatibility)
        """
        # All three embeddings are the same unified embedding, so we just store one
        self.unified_history_dq.append(state_embedding)
    
class Encoder(nn.Module):
    def __init__(self, state_use_self_attention, conv_size=4, embed_dim=256, num_heads=8, hidden_dim=256, fusion_method='weighted_sum'):
        super(Encoder, self).__init__()
        #Arguments prepare
        self.use_self_attention = state_use_self_attention
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        conv_output_dim = embed_dim * (6 - conv_size + 1)  # 对于conv3和conv4

        #network4StateEncoder - 6 state features
        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # last bitrate
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # current buffer size
        self.conv3 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k throughput
        self.conv4 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k download time
        self.proj_conv3 = nn.Linear(conv_output_dim, embed_dim)   # embed_dim * (6 - conv_size + 1) -> embed_dim
        self.proj_conv4 = nn.Linear(conv_output_dim, embed_dim)   # embed_dim * (6 - conv_size + 1) -> embed_dim
        self.conv5 = nn.Sequential(nn.Conv1d(1, embed_dim, BITRATE_LEVELS), nn.LeakyReLU(), nn.Flatten())  # next chunk sizes
        self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # remain chunks  

        #network4Action and Return - 2 additional features
        self.fc_action = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # action
        self.fc_return = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # return

        # Self-attention layer that fuses 8 features (6 state + 1 action + 1 return) into a single hidden representation
        if self.use_self_attention:
            self.feature_attention = FeatureSelfAttention(
                embed_dim=embed_dim, 
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                num_features=8,  # 6 state features + 1 action + 1 return
                fusion_method=fusion_method
            )


    def forward(self, state, action, return_val):
        """
        Args:
            state: (batch_size, seq_len, 6, 6) - network states
            action: (batch_size, seq_len, 1) - actions (bitrate levels)
            return_val: (batch_size, seq_len, 1) - returns (cumulative rewards)
        
        Returns:
            hidden_representation: (batch_size, seq_len, hidden_dim) - unified hidden representation
        """
        # state.shape: (batch_size, seq_len, 6, 6) -> (batch_size x seq_len, 6, 6)
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, 6, 6)
        
        # Process state features (6 features)
        last_bitrate = state[..., 0:1, -1]
        current_buffer_size = state[..., 1:2, -1]
        throughputs = state[..., 2:3, :]
        download_time = state[..., 3:4, :]
        next_chunk_size = state[..., 4:5, :BITRATE_LEVELS]
        remain_chunks = state[..., 5:6, -1]

        features1 = self.fc1(last_bitrate).reshape(batch_size, seq_len, -1)
        features2 = self.fc2(current_buffer_size).reshape(batch_size, seq_len, -1)
        features3 = self.conv3(throughputs).reshape(batch_size, seq_len, -1)
        features3 = self.proj_conv3(features3)
        features4 = self.conv4(download_time).reshape(batch_size, seq_len, -1)
        features4 = self.proj_conv4(features4)
        features5 = self.conv5(next_chunk_size).reshape(batch_size, seq_len, -1)
        features6 = self.fc6(remain_chunks).reshape(batch_size, seq_len, -1)

        # Process action and return features
        action = action.to(self.fc_action[0].weight.device).float()  # Ensure correct device and dtype
        return_val = return_val.to(self.fc_return[0].weight.device).float()
        
        features7 = self.fc_action(action)  # (batch_size, seq_len, embed_dim)
        features8 = self.fc_return(return_val)  # (batch_size, seq_len, embed_dim)

        if self.use_self_attention:
            # Self-attention fuses 8 features (6 state + 1 action + 1 return) into a single hidden representation
            hidden_representation = self.feature_attention([features1, features2, features3, features4, features5, features6, features7, features8])
            # Return as a single tensor: (batch_size, seq_len, hidden_dim)
            return hidden_representation
        else:
            # Without self-attention, concatenate all features
            all_features = torch.cat([features1, features2, features3, features4, features5, features6, features7, features8], dim=-1)
            # Project to hidden_dim
            if not hasattr(self, 'concat_proj'):
                self.concat_proj = nn.Linear(self.embed_dim * 8, self.hidden_dim).to(all_features.device)
            return self.concat_proj(all_features)

class FeatureSelfAttention(nn.Module):
    """
    Self-attention layer for modeling relationships between multiple features (state + action + return).
    After self-attention, all features are fused into a single hidden representation.
    """
    def __init__(self, embed_dim, hidden_dim=None, num_heads=8, num_features=8, dropout=0.1, fusion_method='weighted_sum'):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features  # 8 features: 6 state + 1 action + 1 return
        self.fusion_method = fusion_method  # 'weighted_sum', 'mean', 'concat'
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Fusion layer: fuse num_features enhanced features into a single hidden representation
        if fusion_method == 'weighted_sum':
            # Learnable weights for each feature
            self.fusion_weights = nn.Parameter(torch.ones(num_features) / num_features)
            self.fusion_proj = nn.Linear(embed_dim, self.hidden_dim)
        elif fusion_method == 'mean':
            self.fusion_proj = nn.Linear(embed_dim, self.hidden_dim)
        elif fusion_method == 'concat':
            self.fusion_proj = nn.Linear(embed_dim * num_features, self.hidden_dim)
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
        
        self.fusion_activation = nn.LeakyReLU()
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, features):
        """
        Args:
            features: List of num_features feature tensors, each with shape (batch_size, seq_len, embed_dim)
        Returns:
            hidden_representation: Single fused hidden representation with shape (batch_size, seq_len, hidden_dim)
        """
        # Stack features: (batch_size, seq_len, num_features, embed_dim)
        batch_size, seq_len = features[0].shape[:2]
        stacked = torch.stack(features, dim=2)  # (batch_size, seq_len, num_features, embed_dim)
        
        # Reshape for attention: (batch_size * seq_len, num_features, embed_dim)
        stacked = stacked.reshape(batch_size * seq_len, self.num_features, self.embed_dim)
        
        # Multi-head attention
        q = self.q_proj(stacked)  # (batch_size * seq_len, num_features, embed_dim)
        k = self.k_proj(stacked)
        v = self.v_proj(stacked)
        
        # Reshape for multi-head: (batch_size * seq_len, num_features, num_heads, head_dim)
        q = q.view(batch_size * seq_len, self.num_features, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size * seq_len, self.num_features, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size * seq_len, self.num_features, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size * seq_len, num_heads, num_features, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size * seq_len, num_features, num_heads, head_dim)
        attn_output = attn_output.view(batch_size * seq_len, self.num_features, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        enhanced_features = self.layer_norm(output + stacked)  # (batch_size * seq_len, num_features, embed_dim)
        
        # Fuse num_features enhanced features into a single hidden representation
        if self.fusion_method == 'weighted_sum':
            # Weighted sum of features
            weights = F.softmax(self.fusion_weights, dim=0)  # (num_features,)
            fused = torch.sum(enhanced_features * weights.unsqueeze(0).unsqueeze(-1), dim=1)  # (batch_size * seq_len, embed_dim)
        elif self.fusion_method == 'mean':
            # Mean pooling
            fused = torch.mean(enhanced_features, dim=1)  # (batch_size * seq_len, embed_dim)
        elif self.fusion_method == 'concat':
            # Concatenate all features
            fused = enhanced_features.view(batch_size * seq_len, self.num_features * self.embed_dim)  # (batch_size * seq_len, embed_dim * num_features)
        
        # Project to hidden dimension
        hidden = self.fusion_proj(fused)  # (batch_size * seq_len, hidden_dim)
        hidden = self.fusion_activation(hidden)
        hidden = self.fusion_norm(hidden)
        
        # Reshape back to (batch_size, seq_len, hidden_dim)
        hidden = hidden.view(batch_size, seq_len, self.hidden_dim)
        
        return hidden

class AlignmentLayer(nn.Module):
    def __init__(self, state_dim, num_heads, key_dim, llm_dim, attention_dropout=0.1):
        super(AlignmentLayer, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.llm_dim = llm_dim
        
        #network4AlignmentLayer
        self.q_proj = nn.Linear(state_dim, key_dim * num_heads)
        self.k_proj = nn.Linear(llm_dim, key_dim * num_heads)
        self.v_proj = nn.Linear(llm_dim, key_dim * num_heads)
        self.out_proj = nn.Linear(key_dim * num_heads, llm_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v):
        S = k.shape[0]
        H = self.num_heads
        B, L, _ = q.shape

        q_embeddings = self.q_proj(q).view(B, L, H, -1)
        k_embeddings = self.k_proj(k).view(S, H, -1)
        v_embeddings = self.v_proj(v).view(S, H, -1)
        cross_attn_embeddings = self.cross_attention(q_embeddings, k_embeddings, v_embeddings)
        cross_attn_embeddings = cross_attn_embeddings.reshape(B, L, -1)
        output = self.out_proj(cross_attn_embeddings)
        return output

    def cross_attention(self, q_embeddings, k_embeddings, v_embeddings):
        scale = 1. / (q_embeddings.size(-1) ** 0.5)
        scores = torch.einsum('blhe,she->bhls', q_embeddings, k_embeddings)
        attn = self.attention_dropout(torch.softmax(scale * scores, dim=-1))  # Fixed: use attention_dropout
        cross_attn_embeddings = torch.einsum('bhls,she->blhe', attn, v_embeddings)
        return cross_attn_embeddings # (batch_size, seq_len, num_heads, head_dim)

if __name__ == "__main__":
    #Print Pickle files
    with open('artifacts/exp_pools/exp_pool.pkl', 'rb') as f:
        exp_pool = pickle.load(f)
    print(dir(exp_pool))
    print("经验池大小:", len(exp_pool))
    # print("动作:", len(exp_pool.actions))
    # 打印前 5 条经验（假设每个字段是列表）
    for i in range(1):
        print(f"经验 {i+1}:")
        print("状态:", exp_pool.states[i])
        print("动作:", exp_pool.actions[i])
        print("奖励:", exp_pool.rewards[i])
        print("是否结束:", exp_pool.dones[i])
        print("-" * 40)
    