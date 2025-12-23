import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
from collections import deque
from plm_special.utils.plm_utils import load_plm_llama, load_plm_qwen3
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from baseline_special.utils.constants import BITRATE_LEVELS
import pickle
from munch import Munch
from plm_special.data.dataset import ExperienceDataset
from pprint import pprint
from plm_special.models.low_rank import peft_model

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
        #self.plm, self.tokenizer, self.plm_config = load_plm_llama(args.model_path)
        self.plm, self.tokenizer, self.plm_config = load_plm_qwen3(args.model_path)

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
        self.action_embedding = nn.Linear(1, self.llm_dim).to(self.device)
        self.return_embedding = nn.Linear(1, self.llm_dim).to(self.device)
        self.timestep_embedding = nn.Embedding(100, self.llm_dim).to(self.device)
        self.mapping_layer = nn.Linear(self.vocabulary_size, self.tiny_vocab_size).to(self.device)
        
        # Projection layers for action and return to state_embedding_dim (for state encoder)
        self.action_proj_to_state_dim = nn.Linear(self.llm_dim, self.state_embedding_dim).to(self.device)
        self.return_proj_to_state_dim = nn.Linear(self.llm_dim, self.state_embedding_dim).to(self.device)
        
        # Projection layers from hidden_dim to llm_dim (for action and return after state encoder)
        if self.state_use_self_attention:
            self.action_proj_to_llm_dim = nn.Linear(self.state_attn_hidden_dim, self.llm_dim).to(self.device)
            self.return_proj_to_llm_dim = nn.Linear(self.state_attn_hidden_dim, self.llm_dim).to(self.device)
            alignment_input_dim = self.state_attn_hidden_dim
        else:
            alignment_input_dim = self.state_embedding_dim * 6
        
        self.state_encoder = StateEncoder(
            self.state_use_self_attention, 
            embed_dim=self.state_embedding_dim,
            hidden_dim=self.state_attn_hidden_dim,
            fusion_method=self.fusion_method
        ).to(self.device)
        self.alignment_layer = AlignmentLayer(alignment_input_dim, args.num_heads, args.key_dim, self.llm_dim).to(self.device)
        self.action_projection = nn.Linear(self.llm_dim, BITRATE_LEVELS).to(self.device)
        
        # modules_except_plm: used for saving/loading modules except PLM (for compatibility with training framework)
        # Note: action_proj_to_llm_dim and return_proj_to_llm_dim are no longer used (alignment_layer handles the conversion)
        # but we keep them in the model for potential future use
        modules_list = [
            self.state_encoder,
            self.action_embedding,
            self.return_embedding,
            self.timestep_embedding,
            self.mapping_layer,
            self.alignment_layer,
            self.action_projection,
            self.action_proj_to_state_dim,
            self.return_proj_to_state_dim
        ]
        if self.state_use_self_attention:
            # Keep these layers even though not currently used, in case needed for future modifications
            modules_list.extend([self.action_proj_to_llm_dim, self.return_proj_to_llm_dim])
        self.modules_except_plm = nn.ModuleList(modules_list)
        
        # Store historical state embeddings, action embeddings, and return embeddings
        # Use float32 for consistency (will convert to bfloat16 when needed for LLM)
        self.states_dq = deque([torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device)], maxlen=self.max_length)
        self.actions_dq = deque([torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device)], maxlen=self.max_length)
        self.returns_dq = deque([torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device)], maxlen=self.max_length)

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        Forward function for training.
        
        Args:
            states: (batch_size, seq_len, 6, 6) - network states
            actions: (batch_size, seq_len, 1) - actions (bitrate levels)
            returns: (batch_size, seq_len, 1) - returns (cumulative rewards)
            timesteps: (batch_size, seq_len) - timesteps
        
        Returns:
            action_pred: (batch_size, seq_len, BITRATE_LEVELS) - predicted action logits
        """
        # Move inputs to device and ensure correct dtype
        # Note: Most modules use float32, only LLM uses bfloat16
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

        # Embed actions and returns for state encoder (project to state_embedding_dim)
        # First embed to llm_dim, then project to state_embedding_dim for state encoder
        action_emb_llm = self.action_embedding(actions)  # (batch, seq_len, llm_dim)
        return_emb_llm = self.return_embedding(returns)  # (batch, seq_len, llm_dim)
        
        # Project to state_embedding_dim for state encoder
        action_emb_for_state = self.action_proj_to_state_dim(action_emb_llm)  # (batch, seq_len, state_embedding_dim)
        return_emb_for_state = self.return_proj_to_state_dim(return_emb_llm)  # (batch, seq_len, state_embedding_dim)
        
        # State encoding with action and return embeddings
        encoder_output = self.state_encoder(states, action_embedding=action_emb_for_state, return_embedding=return_emb_for_state)
        
        # Handle different return formats
        if self.state_use_self_attention:
            # Returns: state_emb, action_emb, return_emb (all in hidden_dim)
            state_emb, action_emb, return_emb = encoder_output
            
            # Align state, action, return embeddings to LLM space (while still in hidden_dim)
            state_embeddings = self.alignment_layer(state_emb, tiny_vocab_embeddings, tiny_vocab_embeddings)  # (batch, seq_len, llm_dim)
            action_embeddings = self.alignment_layer(action_emb, tiny_vocab_embeddings, tiny_vocab_embeddings)  # (batch, seq_len, llm_dim)
            return_embeddings = self.alignment_layer(return_emb, tiny_vocab_embeddings, tiny_vocab_embeddings)  # (batch, seq_len, llm_dim)
        else:
            # Returns: tuple of 6 features
            states_features = encoder_output
            states_features_concat = torch.cat(states_features, dim=-1)
            state_embeddings = self.alignment_layer(states_features_concat, tiny_vocab_embeddings, tiny_vocab_embeddings)
            # For backward compatibility, use original embeddings
            action_embeddings = action_emb_llm
            return_embeddings = return_emb_llm
        
        timestep_embeddings = self.timestep_embedding(timesteps)  # (batch, seq_len, llm_dim)
        
        # Add timestep embeddings (similar to positional embeddings)
        state_embeddings = state_embeddings + timestep_embeddings
        action_embeddings = action_embeddings + timestep_embeddings
        return_embeddings = return_embeddings + timestep_embeddings

        # Concatenate instruction and state features
        # Format: [instruction, (return, state, action) for each timestep]
        # Note: All embeddings are in float32 for consistency with LoRA layers
        concated_embeddings = [instruction_embeddings]  # LLM embeddings are float32
        
        #test without prefix prompt
        #concated_embeddings = []

        # Stack returns, states, actions for each timestep
        batch_size, seq_len = states.shape[0], states.shape[1]
        for i in range(seq_len):
            # All embeddings are already in float32
            concated_embeddings.append(return_embeddings[:, i:i+1, :])  # (batch, 1, llm_dim)
            concated_embeddings.append(state_embeddings[:, i:i+1, :])  # (batch, 1, llm_dim)
            concated_embeddings.append(action_embeddings[:, i:i+1, :])  # (batch, 1, llm_dim)
        
        concated_embeddings = torch.cat(concated_embeddings, dim=1)  # (batch, total_seq_len, llm_dim), dtype: float32

        # LLM forward
        output = self.plm(inputs_embeds=concated_embeddings).last_hidden_state
        # output shape: (batch, total_seq_len, llm_dim)
        
        # Extract outputs corresponding to state positions for action prediction
        # State positions are at: instruction_len + 1, instruction_len + 4, instruction_len + 7, ...
        instruction_len = instruction_embeddings.shape[1]
        state_positions = [instruction_len + 1 + i * 3 for i in range(seq_len)]  # Positions of state embeddings
        state_outputs = output[:, state_positions, :]  # (batch, seq_len, llm_dim), dtype: float32
        
        # Predict actions
        # state_outputs is already in float32
        action_pred = self.action_projection(state_outputs)  # (batch, seq_len, BITRATE_LEVELS), dtype: float32
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
        
        # Step 4: Process target return and timestep
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep_tensor = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)
        
        return_emb_llm = self.return_embedding(target_return)  # (1, 1, llm_dim)
        timestep_embeddings = self.timestep_embedding(timestep_tensor)  # (1, 1, llm_dim)
        
        # Project return to state_embedding_dim for state encoder
        return_emb_for_state = self.return_proj_to_state_dim(return_emb_llm)  # (1, 1, state_embedding_dim)
        
        # Step 5: Process state with return embedding (no action yet, as we're predicting it)
        state = state.to(self.device).float()  # (1, 1, 6, 6)
        encoder_output = self.state_encoder(state, action_embedding=None, return_embedding=return_emb_for_state)
        
        # Handle different return formats
        if self.state_use_self_attention:
            # Returns: state_emb, action_emb (None), return_emb (all in hidden_dim)
            state_emb, _, return_emb = encoder_output
            
            # Align state and return embeddings to LLM space (while still in hidden_dim)
            state_embeddings = self.alignment_layer(state_emb, tiny_vocab_embeddings, tiny_vocab_embeddings)  # (1, 1, llm_dim)
            return_embeddings = self.alignment_layer(return_emb, tiny_vocab_embeddings, tiny_vocab_embeddings)  # (1, 1, llm_dim)
        else:
            # Returns: tuple of 6 features
            states_features = encoder_output
            states_features_concat = torch.cat(states_features, dim=-1)
            state_embeddings = self.alignment_layer(states_features_concat, tiny_vocab_embeddings, tiny_vocab_embeddings)
            return_embeddings = return_emb_llm
        
        # Add timestep embeddings
        state_embeddings = state_embeddings + timestep_embeddings  # (1, 1, llm_dim)
        return_embeddings = return_embeddings + timestep_embeddings  # (1, 1, llm_dim)
        
        # Step 6: Concatenate: [historical, instruction, return, state]
        # Format matches forward: [instruction, (return, state, action) for each timestep]
        # For sample: [historical(return, state, action)..., instruction, return, state]
        concated_embeddings = []
        
        # Add historical context if exists (already in float32)
        if prev_stacked_inputs.shape[1] > 0:
            concated_embeddings.append(prev_stacked_inputs)
        
        # Add instruction (already in float32)
        concated_embeddings.append(instruction_embeddings)
        
        # Add current return, state (already in float32)
        concated_embeddings.append(return_embeddings)
        concated_embeddings.append(state_embeddings)
        
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
        
        # Step 8: Predict action from the last state position
        # The last position corresponds to the state embedding we just added
        logits_used = output[:, -1:, :]  # (1, 1, llm_dim) - use last position (state embedding), dtype: float32
        action_pred = self.action_projection(logits_used)  # (1, 1, BITRATE_LEVELS), dtype: float32
        action_pred = action_pred.reshape(-1)  # (BITRATE_LEVELS,)
        
        # Sample action
        bitrate, _ = self._sample(action_pred)
        
        # Step 9: Compute action embeddings for history
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = (bitrate + 1) / BITRATE_LEVELS
        action_embeddings = self.action_embedding(action_tensor) + timestep_embeddings  # (1, 1, llm_dim)
        
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
        Clear all deques and reset them to empty state.
        This should be called at the start of each new episode.
        """
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        
        # Reinitialize with empty tensors (use float32 for consistency with embedding outputs)
        self.states_dq.append(torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device))
    
    def get_historical_context(self):
        """
        Get historical context from deques for building sequence input.
        Returns concatenated historical embeddings.
        
        Returns:
            historical_context: (1, total_seq_len, llm_dim) - concatenated historical embeddings
        """
        # Stack historical embeddings: (return, state, action) for each time step
        historical_inputs = []
        for i in range(len(self.states_dq)):
            prev_return = self.returns_dq[i]
            prev_state = self.states_dq[i]
            prev_action = self.actions_dq[i]
            # Check if any of them has non-zero sequence length
            if prev_return.shape[1] > 0 or prev_state.shape[1] > 0 or prev_action.shape[1] > 0:
                # Concatenate: (return, state, action) for each historical step
                historical_inputs.append(torch.cat((prev_return, prev_state, prev_action), dim=1))
        
        if len(historical_inputs) > 0:
            return torch.cat(historical_inputs, dim=1)
        else:
            # Return empty tensor if no history (use float32 for consistency)
            return torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device)
    
    def update_history(self, state_embedding, action_embedding, return_embedding):
        """
        Update history deques with new embeddings.
        
        Args:
            state_embedding: (1, seq_len, llm_dim) - state embedding after alignment
            action_embedding: (1, seq_len, llm_dim) - action embedding
            return_embedding: (1, seq_len, llm_dim) - return embedding (optional, can be None)
        """
        self.states_dq.append(state_embedding)
        self.actions_dq.append(action_embedding)
        if return_embedding is not None:
            self.returns_dq.append(return_embedding)
        else:
            # Append empty tensor if return_embedding is not provided (use float32 for consistency)
            self.returns_dq.append(torch.zeros((1, 0, self.llm_dim), dtype=torch.float32, device=self.device))

    # def get_tokenizer_size(self):
    #     return self.word_embeddings.shape
    # def get_state_size(self):
    #     a=torch.rand(1, 10, 6, 6)
    #     features = self.state_encoder(a)
    #     return features[0].shape, features[1].shape, features[2].shape, features[3].shape, features[4].shape, features[5].shape
    # def get_timestep_size(self):
    #     timestep = torch.randint(0, 100, (1, 10), dtype=torch.long, device=self.device)
    #     timestep_embeddings = self.timestep_embedding(timestep)
    #     return timestep_embeddings.shape
    # def get_instruction_size(self):
    #     instruction = (
    #         f"<|start_prompt|>Data description: {self.data_description}"
    #         f" Task description: {self.task_description} "
    #         f" Input statistics: "
    #         f"<|end_prompt|>"
    #     )
    #     instruction_tokens = self.tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
    #     instruction_tokens = instruction_tokens.to(self.device)
    #     instruction_embeddings = self.plm.get_input_embeddings()(instruction_tokens)
    #     return instruction_embeddings.shape
    # def get_action_size(self):
    #     action = torch.rand((1, 10, 1), dtype=torch.float32, device=self.device)
    #     action_embeddings = self.action_embedding(action)
    #     return action_embeddings.shape
    # def test_forward(self):
    #     batch_size = 1
    #     seq_len = 10
    #     states = torch.rand((batch_size, seq_len, 6, 6), dtype=torch.float32)
    #     actions = torch.randint(0, BITRATE_LEVELS, (batch_size, seq_len, 1), dtype=torch.float32)
    #     returns = torch.rand((batch_size, seq_len, 1), dtype=torch.float32)
    #     timesteps = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
    #     action_pred = self.forward(states, actions, returns, timesteps)
    #     print("Action prediction shape:", action_pred.shape)  # Expected: (batch_size, seq_len, BITRATE_LEVELS)
    
class StateEncoder(nn.Module):
    def __init__(self, state_use_self_attention, conv_size=4, embed_dim=256, num_heads=8, hidden_dim=256, fusion_method='weighted_sum'):
        super(StateEncoder, self).__init__()
        #Arguments prepare
        self.use_self_attention = state_use_self_attention
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        conv_output_dim = embed_dim * (6 - conv_size + 1)  # 对于conv3和conv4

        #network4StateEncoder
        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # last bitrate
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # current buffer size
        self.conv3 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k throughput
        self.conv4 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k download time
        self.proj_conv3 = nn.Linear(conv_output_dim, embed_dim)   # embed_dim * (6 - conv_size + 1) -> embed_dim
        self.proj_conv4 = nn.Linear(conv_output_dim, embed_dim)   # embed_dim * (6 - conv_size + 1) -> embed_dim
        self.conv5 = nn.Sequential(nn.Conv1d(1, embed_dim, BITRATE_LEVELS), nn.LeakyReLU(), nn.Flatten())  # next chunk sizes
        self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # remain chunks  

        # Self-attention layer that fuses 6 features into a single hidden representation
        if self.use_self_attention:
            self.state_attention = StateFeatureSelfAttention(
                embed_dim=embed_dim, 
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                fusion_method=fusion_method
            )


    def forward(self, state, action_embedding=None, return_embedding=None):
        """
        Args:
            state: (batch_size, seq_len, 6, 6) - network states
            action_embedding: (batch_size, seq_len, embed_dim) or None - action embeddings
            return_embedding: (batch_size, seq_len, embed_dim) or None - return embeddings
        Returns:
            If use_self_attention:
                state_emb, action_emb, return_emb - separate embeddings for each
            Else:
                features1, features2, features3, features4, features5, features6 - 6 separate state features
        """
        # state.shape: (batch_size, seq_len, 6, 6) -> (batch_size x seq_len, 6, 6)
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, 6, 6)
        
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

        if self.use_self_attention:
            # Causal masked self-attention with state, action, return
            state_emb, action_emb, return_emb = self.state_attention(
                [features1, features2, features3, features4, features5, features6],
                action_embedding=action_embedding,
                return_embedding=return_embedding
            )
            # Return separate embeddings
            return state_emb, action_emb, return_emb
        
        # Without self-attention, return 6 separate features (for backward compatibility)
        return features1, features2, features3, features4, features5, features6

class StateFeatureSelfAttention(nn.Module):
    """
    Causal masked self-attention layer for modeling relationships between state, action, and return.
    Masking rules:
    - State features (0-5): can only attend to state features (0-5)
    - Action (6): can attend to state features (0-5) and action (6)
    - Return (7): can attend to state features (0-5), action (6), and return (7)
    
    Returns separate embeddings for state, action, and return.
    """
    def __init__(self, embed_dim, hidden_dim=None, num_heads=8, dropout=0.1, fusion_method='weighted_sum'):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fusion_method = fusion_method  # 'weighted_sum', 'mean', 'concat'
        self.num_state_features = 6  # 6 state features
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Separate output projections for state, action, and return
        self.state_fusion_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.action_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.return_proj = nn.Linear(embed_dim, self.hidden_dim)
        
        # Fusion layer for state features
        if fusion_method == 'weighted_sum':
            self.fusion_weights = nn.Parameter(torch.ones(6) / 6)
        elif fusion_method == 'mean':
            pass  # No weights needed
        elif fusion_method == 'concat':
            self.state_fusion_proj = nn.Linear(embed_dim * 6, self.hidden_dim)
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
        
        self.fusion_activation = nn.LeakyReLU()
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        
        # Create causal mask (will be created in forward)
        self.register_buffer('causal_mask', None)

    def _create_causal_mask(self, seq_len, device):
        """
        Create causal mask for state, action, return.
        Total length: 8 (6 state features + 1 action + 1 return)
        Mask rules:
        - State (0-5): can see state (0-5)
        - Action (6): can see state (0-5) and action (6)
        - Return (7): can see state (0-5), action (6), and return (7)
        """
        mask = torch.zeros(8, 8, device=device, dtype=torch.bool)
        
        # State features (0-5) can only see state features (0-5)
        mask[:6, :6] = True
        
        # Action (6) can see state (0-5) and action (6)
        mask[6, :7] = True
        
        # Return (7) can see state (0-5), action (6), and return (7)
        mask[7, :] = True
        
        return mask

    def forward(self, state_features, action_embedding=None, return_embedding=None):
        """
        Args:
            state_features: List of 6 feature tensors, each with shape (batch_size, seq_len, embed_dim)
            action_embedding: (batch_size, seq_len, embed_dim) or None
            return_embedding: (batch_size, seq_len, embed_dim) or None
        Returns:
            state_emb: (batch_size, seq_len, hidden_dim) - state embedding
            action_emb: (batch_size, seq_len, hidden_dim) or None - action embedding
            return_emb: (batch_size, seq_len, hidden_dim) or None - return embedding
        """
        batch_size, seq_len = state_features[0].shape[:2]
        device = state_features[0].device
        
        # Stack state features: (batch_size, seq_len, 6, embed_dim)
        state_stacked = torch.stack(state_features, dim=2)  # (batch_size, seq_len, 6, embed_dim)
        
        # Prepare all features: state (6) + action (1) + return (1) = 8 total
        all_features = []
        
        # Reshape for attention: (batch_size * seq_len, 6, embed_dim)
        state_reshaped = state_stacked.reshape(batch_size * seq_len, 6, self.embed_dim)
        all_features.append(state_reshaped)
        
        if action_embedding is not None:
            action_reshaped = action_embedding.reshape(batch_size * seq_len, 1, self.embed_dim)
            all_features.append(action_reshaped)
        
        if return_embedding is not None:
            return_reshaped = return_embedding.reshape(batch_size * seq_len, 1, self.embed_dim)
            all_features.append(return_reshaped)
        
        # Concatenate all features: (batch_size * seq_len, total_features, embed_dim)
        stacked = torch.cat(all_features, dim=1)  # (batch_size * seq_len, 6/7/8, embed_dim)
        total_features = stacked.shape[1]
        
        # Create causal mask
        causal_mask = self._create_causal_mask(total_features, device)
        # Only use the relevant part of the mask
        causal_mask = causal_mask[:total_features, :total_features]
        
        # Multi-head attention
        q = self.q_proj(stacked)  # (batch_size * seq_len, total_features, embed_dim)
        k = self.k_proj(stacked)
        v = self.v_proj(stacked)
        
        # Reshape for multi-head: (batch_size * seq_len, total_features, num_heads, head_dim)
        q = q.view(batch_size * seq_len, total_features, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size * seq_len, total_features, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size * seq_len, total_features, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size * seq_len, num_heads, total_features, total_features)
        
        # Apply causal mask: set masked positions to -inf
        mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total_features, total_features)
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size * seq_len, num_heads, total_features, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size * seq_len, total_features, num_heads, head_dim)
        attn_output = attn_output.view(batch_size * seq_len, total_features, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        enhanced_features = self.layer_norm(output + stacked)  # (batch_size * seq_len, total_features, embed_dim)
        
        # Extract state, action, return embeddings
        state_enhanced = enhanced_features[:, :6, :]  # (batch_size * seq_len, 6, embed_dim)
        
        # Fuse state features
        if self.fusion_method == 'weighted_sum':
            weights = F.softmax(self.fusion_weights, dim=0)  # (6,)
            state_fused = torch.sum(state_enhanced * weights.unsqueeze(0).unsqueeze(-1), dim=1)  # (batch_size * seq_len, embed_dim)
        elif self.fusion_method == 'mean':
            state_fused = torch.mean(state_enhanced, dim=1)  # (batch_size * seq_len, embed_dim)
        elif self.fusion_method == 'concat':
            state_fused = state_enhanced.view(batch_size * seq_len, 6 * self.embed_dim)  # (batch_size * seq_len, embed_dim * 6)
        
        # Project to hidden dimension
        state_emb = self.state_fusion_proj(state_fused)  # (batch_size * seq_len, hidden_dim)
        state_emb = self.fusion_activation(state_emb)
        state_emb = self.fusion_norm(state_emb)
        state_emb = state_emb.view(batch_size, seq_len, self.hidden_dim)
        
        # Extract action and return embeddings if they exist
        action_emb = None
        return_emb = None
        
        if action_embedding is not None:
            action_enhanced = enhanced_features[:, 6, :]  # (batch_size * seq_len, embed_dim)
            action_emb = self.action_proj(action_enhanced)  # (batch_size * seq_len, hidden_dim)
            action_emb = self.fusion_activation(action_emb)
            action_emb = action_emb.view(batch_size, seq_len, self.hidden_dim)
        
        if return_embedding is not None:
            return_idx = 7 if action_embedding is not None else 6
            return_enhanced = enhanced_features[:, return_idx, :]  # (batch_size * seq_len, embed_dim)
            return_emb = self.return_proj(return_enhanced)  # (batch_size * seq_len, hidden_dim)
            return_emb = self.fusion_activation(return_emb)
            return_emb = return_emb.view(batch_size, seq_len, self.hidden_dim)
        
        return state_emb, action_emb, return_emb

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

def print_pickle_files():
    with open('artifacts/exp_pools/exp_pool.pkl', 'rb') as f:
        exp_pool = pickle.load(f)
    print(dir(exp_pool))
    print("经验池大小:", len(exp_pool.states))
    # print("动作:", len(exp_pool.actions))
    # 打印前 5 条经验（假设每个字段是列表）
    for i in range(50):
        print(f"经验 {i+1}:")
        print("状态:", exp_pool.states[i])
        print("动作:", exp_pool.actions[i])
        print("奖励:", exp_pool.rewards[i])
        print("是否结束:", exp_pool.dones[i])
        print("-" * 40)
    
def prepare_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)    
    args = parser.parse_args()

    args.adapt = True
    args.test = True
    args.grad_accum_steps = 32
    args.seed = 666
    args.scale = 1000
    args.model_dir = None
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.device_out = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.device_mid = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.plm_type = 'llama'
    args.plm_size = 'large'
    args.rank =128
    args.state_feature_dim = 256
    args.state_embedding_dim = 256
    args.llm_dim = 3072

    args.frozen = True
    args.num_heads = 8
    args.key_dim = 128
    args.state_use_self_attention = True
    args.state_attn_hidden_dim = 256
    args.fusion_method = 'weighted_sum'

    args.w = 20
    args.max_length = args.w
    args.gamma = 1.
    args.lr = 1e-4
    args.weight_decay = 1e-4
    args.warmup_steps = 2000
    args.num_epochs = 50
    args.eval_per_epoch = 2
    args.save_checkpoint_per_epoch = 10
    args.target_return_scale = 1.
    args.which_layer = -1

    args.exp_pool_path = 'artifacts/exp_pools/exp_pool.pkl'
    args._base_dir = '' if 'adaptive_bitrate_streaming' in os.getcwd() else 'adaptive_bitrate_streaming/'
    args.plm_dir = args._base_dir + ('../../downloaded_plms' if 'adaptive_bitrate_streaming' in args._base_dir else '../downloaded_plms')
    args.model_path = os.path.join(args.plm_dir, args.plm_type, args.plm_size)
    args.sample_step = None
    args.trace = 'fcc-test'
    args.trace_num = 100
    args.video = 'video1'
    args.fixed_order = True

    return args

def test_save_model():
    args = prepare_args()
    abrllm_model = ABRLLM(args)
    abrllm_model.device = torch.device(args.device)
    abrllm_model = abrllm_model.to(args.device)
    abrllm_model.plm = peft_model(abrllm_model.plm, args.plm_type, rank=args.rank)
    print(abrllm_model)

    # plm_ft_dir = args._base_dir + 'data/ft_plms'
    # train_exp_pool_info = args.exp_pool_path.split('/')[-4:-1]
    # train_exp_pool_info = '_'.join(train_exp_pool_info)
    # models_dir = os.path.join(
    #     plm_ft_dir, 
    #     f'{args.plm_type}_{args.plm_size}', 
    #     train_exp_pool_info + f'_ss_{args.sample_step}', 
    #     f'abrllm_rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_sfd_{args.state_feature_dim}'
    #     f'_sattn_{args.state_use_self_attention}_sahd_{args.state_attn_hidden_dim}_fusion_{args.fusion_method}'
    #     f'_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}_seed_{args.seed}'
    # )
    # checkpoint_dir = os.path.join(models_dir, 'checkpoint')
    # best_model_dir = os.path.join(models_dir, 'best_model')
    # save_dir = os.path.join(checkpoint_dir, str(0))
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # if args.rank > 0:
    #     # save lora weights
    #     abrllm_model.plm.save_pretrained(save_dir)
    #     # save other modules except plm
    #     torch.save(abrllm_model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))

if __name__ == "__main__":
    test_save_model()
    