import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
import binascii
from urllib import parse
from utils import load_plm_llama, load_plm_qwen3

class SocketEncoder(nn.Module):
    def __init__(self, llm_dim, hidden_dim=64):
        super(SocketEncoder, self).__init__()
        self.ip_embedding = nn.Embedding(256, hidden_dim)
        self.port_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU()
        )
        fusion_input_dim = hidden_dim * (4 + 1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(), # 使用 GELU 激活函数通常比 ReLU 效果更好
            nn.Linear(1024, llm_dim)
        )
        self.residual_connection = nn.Linear(fusion_input_dim, llm_dim)
        self.final_ln = nn.LayerNorm(llm_dim)

    def _encode_single_socket(self, ip, port):
        # ip_tensor shape: (batch_size, 4)
        # port_tensor shape: (batch_size, 1)
        # CRITICAL: Ensure port is float type for Linear layer
        # Force conversion to float32 to avoid dtype mismatch
        port = port.to(torch.float32)
        ip_features = self.ip_embedding(ip).flatten(1)  # (batch_size, 4 * hidden_dim)
        port_features = self.port_embedding(port)  # (batch_size, hidden_dim)
        raw_features = torch.cat([ip_features, port_features], dim=1)  # (batch_size, fusion_input_dim)
        #Output = LayerNorm( MLP(x) + Projection(x) )
        socket_token = self.fusion_mlp(raw_features)  # (batch_size, llm_dim)
        socket_token += self.residual_connection(raw_features)
        socket_token = self.final_ln(socket_token)
        return socket_token

    def forward(self, src_ip, dst_ip, src_port, dst_port):
        src_token = self._encode_single_socket(src_ip, src_port)
        dst_token = self._encode_single_socket(dst_ip, dst_port)
        return src_token, dst_token

class PayloadEncoder(nn.Module):
    def __init__(self, llm_dim, max_len=256, hidden_dim=32):
        super(PayloadEncoder, self).__init__()
        #256 means 0x00~0xff  
        self.byte_embedding = nn.Embedding(256, hidden_dim)
        self.feature_extractor = nn.Sequential(
            # 第一层卷积: 捕捉低级特征 (如 3-gram)
            # Input: (Batch, hidden_dim, max_len)
            nn.Conv1d(in_channels=hidden_dim, out_channels=64, kernel_size=3, padding=1),
            nn.LayerNorm([64, max_len]), # 对 Channel 和 Length 维度归一化有助于稳定
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 降采样，长度减半 -> max_len/2
            # 第二层卷积: 捕捉高级组合特征
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 再次降采样 -> max_len/4
        )
        #128 channels
        flatten_dim = 128 * (max_len // 4)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, llm_dim),
            nn.LayerNorm(llm_dim) # 这是一个 Token，最后一定要归一化
        )

    def forward(self, payload_indices):
        # payload_indices: 0-255 的字节索引
        # payload shape: (batch_size, max_len)
        x = self.byte_embedding(payload_indices) # (batch_size, max_len, hidden_dim)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, hidden_dim, max_len) 以适应 Conv1d
        features = self.feature_extractor(x)  # (batch_size, 128, max_len/4)
        payload_token = self.projector(features)  # (batch_size, llm_dim)
        return payload_token

class AugmentedPayloadEncoder(nn.Module):
    def __init__(self, llm_dim, max_len=256, hidden_dim=32, top_k_dist=5):
        super(AugmentedPayloadEncoder, self).__init__()
        '''1. CNN Encoder for origin payload'''
        self.cnn_dim = 128
        self.byte_embed = nn.Embedding(256, 32)
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, self.cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Global Max Pooling -> (Batch, 128, 1)
        )
        '''2. MLP Encoder for payload length'''
        self.len_dim = 32
        self.len_mlp = nn.Sequential(
            nn.Linear(1, self.len_dim),
            nn.ReLU(),
            nn.Linear(self.len_dim, self.len_dim)
        )
        '''3. Hamming distance encoder'''
        self.dist_dim = 64
        self.dist_mlp = nn.Sequential(
            nn.Linear(top_k_dist, self.dist_dim),
            nn.ReLU(),
            nn.Linear(self.dist_dim, self.dist_dim)
        )
        '''4. Fusion & Projection'''
        self.fusion_dim = self.cnn_dim + self.len_dim + self.dist_dim
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU()
        )

    def forward(self, payload_seq, payload_len, hamming_dist):
        x = self.byte_embed(payload_seq).permute(0, 2, 1)
        feat_cnn = self.cnn(x).flatten(1)       #(B, 128)
        feat_len = self.len_mlp(payload_len)    #(B, 32)
        feat_dist = self.dist_mlp(hamming_dist) #(B, 64)
        combined = torch.cat([feat_cnn, feat_len, feat_dist], dim=1)
        out = self.fusion(combined)
        return out

class ClassifyPayloadEncoder(nn.Module):
    def __init__(self, llm_dim, hidden_dim=32):
        super(ClassifyPayloadEncoder, self).__init__()
        self.hidden_dim = 32
        self.classify_encoder = nn.Embedding(256, self.hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim * 1, llm_dim),
            nn.LayerNorm(llm_dim)
        )   
    
    def forward(self, payload_head):
        x = self.classify_encoder(payload_head)  # (batch_size, hidden_dim)
        payload_token = self.projector(x)  # (batch_size, llm_dim)
        return payload_token

class HostEncoder(nn.Module):
    def __init__(self, llm_dim, max_len=32, hidden_dim=32):
        super(HostEncoder, self).__init__()
        #128 meaning ASCII range
        self.char_embedding = nn.Embedding(128, hidden_dim)
        self.cnn = nn.Sequential(
            # 捕捉 3-gram (如 'com', 'www')
            nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> len 32
            
            # 捕捉 5-gram (更长的语义块)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Global Max Pooling，无论输入多长都压缩为1
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, llm_dim),
            nn.LayerNorm(llm_dim)
        )

    def forward(self, host_indices):
        # host_indices: 0-127 的字符索引
        # host_indices shape: (batch_size, max_len)
        x = self.char_embedding(host_indices)  # (batch_size, max_len, hidden_dim)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, hidden_dim, max_len)
        features = self.cnn(x)  # (batch_size, 128, 1)
        host_token = self.projector(features)  # (batch_size, llm_dim)
        return host_token   

class AlignmentLayer(nn.Module):
    def __init__(self, input_dim, num_heads, key_dim, llm_dim, attention_dropout=0.1):
        super(AlignmentLayer, self).__init__()
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(input_dim, key_dim * num_heads)
        self.k_proj = nn.Linear(llm_dim, key_dim * num_heads)
        self.v_proj = nn.Linear(llm_dim, key_dim * num_heads)
        self.out_proj = nn.Linear(key_dim * num_heads, llm_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v):
        S = k.shape[0]
        H = self.num_heads
        B, _ = q.shape

        q_embeddings = self.q_proj(q).view(B, H, -1)
        k_embeddings = self.k_proj(k).view(S, H, -1)
        v_embeddings = self.v_proj(v).view(S, H, -1)
        cross_attn_embeddings = self.cross_attention(q_embeddings, k_embeddings, v_embeddings)
        cross_attn_embeddings = cross_attn_embeddings.reshape(B, -1)
        output = self.out_proj(cross_attn_embeddings)
        return output

    def cross_attention(self, q_embeddings, k_embeddings, v_embeddings):
        scale = 1. / (q_embeddings.size(-1) ** 0.5)
        scores = torch.einsum('bhe,she->bhs', q_embeddings, k_embeddings)
        attn = self.attention_dropout(torch.softmax(scale * scores, dim=-1))  # Fixed: use attention_dropout
        cross_attn_embeddings = torch.einsum('bhs,she->bhe', attn, v_embeddings)
        return cross_attn_embeddings # (batch_size, seq_len, num_heads, head_dim)

class IBRLLM(nn.Module):
    def __init__(self, args):
        super(IBRLLM, self).__init__()
        self.args = args
        self.tiny_vocab_size = 1000
        self.feature_dim = 256
        self.llm_dim = args.llm_dim
        self.plm_embed_size = self.llm_dim
        self.payload_max_len = args.payload_max_len
        self.host_max_len = args.host_max_len
        # Ensure device is torch.device type
        self.device = torch.device(args.device) if isinstance(args.device, str) else args.device

        # Load PLM first
        self.plm, self.tokenizer, self.plm_config = load_plm_llama(args.model_path)
        #self.plm, self.tokenizer, self.plm_config = load_plm_qwen3(args.model_path)
        # Move PLM to device immediately to avoid device mismatch
        self.plm = self.plm.to(self.device)
        self.word_embeddings = self.plm.get_input_embeddings().weight #torch.Size([vocab_size, llm_dim])
        self.vocabulary_size = self.word_embeddings.shape[0]

        # Prepare instruction embeddings
        instruction = (
            "Domain: Network traffic analysis. "
            "Input: Payload bytes, IP socket addresses, and Host domains. "
            "Task: Classify the specific application label based on Payload, Source IP, Source Port, Destination IP, Destination Port and Host information. "
        )
        instruction_tokens = self.tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        instruction_tokens = instruction_tokens.to(self.device).long()
        with torch.no_grad():
            self.instruction_embeddings = self.plm.get_input_embeddings()(instruction_tokens)

        # Initialize modules (move to device)
        self.mapping_layer = nn.Linear(self.vocabulary_size, self.tiny_vocab_size).to(self.device)
        self.socket_encoder = SocketEncoder(self.llm_dim).to(self.device)
        #TODO: Try ClassifyPayloadEncoder
        #self.payload_encoder = PayloadEncoder(self.llm_dim, max_len=self.payload_max_len).to(self.device)
        self.payload_encoder = ClassifyPayloadEncoder(self.llm_dim).to(self.device)
        self.host_encoder = HostEncoder(self.llm_dim, max_len=self.host_max_len).to(self.device)
        self.align_socket = AlignmentLayer(input_dim=self.llm_dim, num_heads=8, key_dim=128, llm_dim=self.llm_dim).to(self.device)
        self.align_payload = AlignmentLayer(input_dim=self.llm_dim, num_heads=8, key_dim=128, llm_dim=self.llm_dim).to(self.device)
        self.align_host = AlignmentLayer(input_dim=self.llm_dim, num_heads=8, key_dim=128, llm_dim=self.llm_dim).to(self.device)
        num_classes = getattr(args, 'num_classes', 45)  # Default to 45 if not specified
        self.classifier = nn.Linear(self.llm_dim, num_classes).to(self.device)
        
        # modules_except_plm: used for saving/loading modules except PLM
        self.modules_except_plm = nn.ModuleList([
            self.mapping_layer,
            self.socket_encoder,
            self.payload_encoder,
            self.host_encoder,
            self.align_socket,
            self.align_payload,
            self.align_host,
            self.classifier
        ])

    def forward(self, payload, src_ip, dst_ip, src_port, dst_port, host):
        # Ensure correct dtypes and devices
        payload = payload.to(self.device).long()  # Long for embedding
        src_ip = src_ip.to(self.device).long()  # Long for embedding
        dst_ip = dst_ip.to(self.device).long()  # Long for embedding
        src_port = src_port.to(self.device).float()  # Float for Linear layer
        dst_port = dst_port.to(self.device).float()  # Float for Linear layer
        host = host.to(self.device).long()  # Long for embedding
        
        batch_size = payload.shape[0]
        word_embeddings_float = self.word_embeddings.permute(1, 0).to(torch.float32)
        tiny_vocab_emb = self.mapping_layer(word_embeddings_float).permute(1, 0)  # (1000, llm_dim)

        feature_src_socket, feature_dst_socket = self.socket_encoder(src_ip, dst_ip, src_port, dst_port)
        # TODO: Try ClassifyPayloadEncoder
        # ClassifyPayloadEncoder expects (batch_size,) - extract first byte from payload
        # payload shape: (batch_size, max_payload_len) -> payload_head shape: (batch_size,)
        payload_head = payload[:, 0]  # Extract first byte from each payload
        feature_payload = self.payload_encoder(payload_head)
        feature_host = self.host_encoder(host)
        q_src = feature_src_socket  # (batch_size, llm_dim)
        q_dst = feature_dst_socket # (batch_size, llm_dim)
        q_payload = feature_payload # (batch_size, llm_dim)
        q_host = feature_host # (batch_size, llm_dim)
        
        token_prompt = self.instruction_embeddings.expand(batch_size, -1, -1)  # (batch_size, prompt_len, llm_dim)

        '''try socket w/o align'''
        # token_src = q_src.unsqueeze(1)
        # token_dst = q_dst.unsqueeze(1)
        token_src = self.align_socket(q_src, tiny_vocab_emb, tiny_vocab_emb).unsqueeze(1)  # (batch_size, 1, llm_dim)
        token_dst = self.align_socket(q_dst, tiny_vocab_emb, tiny_vocab_emb).unsqueeze(1)  # (batch_size, 1, llm_dim)

        '''try payload w/o align'''
        #token_payload = self.align_payload(q_payload, tiny_vocab_emb, tiny_vocab_emb).unsqueeze(1)  # (batch_size, 1, llm_dim)
        token_payload = q_payload.unsqueeze(1)  # Directly use payload token without alignment

        '''try host w/o align'''
        #token_host = self.align_host(q_host, tiny_vocab_emb, tiny_vocab_emb).unsqueeze(1)  # (batch_size, 1, llm_dim)
        token_host = q_host.unsqueeze(1)  # Directly use host token without alignment

        llm_input = torch.cat([token_prompt, token_payload, token_src, token_dst, token_host], dim=1)  # (batch_size, total_seq_len, llm_dim)

        outputs = self.plm(inputs_embeds=llm_input)
        last_hidden_state = outputs.last_hidden_state
        logits = self.classifier(last_hidden_state[:, -1, :])
        
        return logits

if __name__ == "__main__":
    from utils import preprocess_payload
    hex = '474554202f643f646e3d393932333062623963643331323339303430376436333134303564663334323526636c69656e7469703d312674746c3d312669643d3120485454502f312e310d0a557365722d4167656e743a 2044616c76696b2f322e312e3020284c696e75783b20553b20416e64726f69642031303b204e583635394a204275696c642f514b51312e3230303430352e303032290d0a486f73743a203138322e3235342e3131362e3131370d0a436f6e6e656374696f6e3a204b6565702d416c6976650d0a4163636570742d456e636f64696e673a20677a69700d0a0d0a485454502f312e3120323030204f4b0d0a436f6e6e656374696f6e3a20636c6f73650d0a5365727665723a2048747470205365727665720d0a436f6e74656e742d547970653a20746578742f68746d6c0d0a436f6e74656e742d4c656e6774683a2039360d0a0d0a386662616533346637353538653061373134623935333863323639636164313836653332333632643237383864663037616262623632383034656166396139626631346537346535633465316431393064323861373039363266653830636230'
    # Test with actual data loading shape: (batch_size, max_payload_len)
    payload_tensor = preprocess_payload(hex, max_len=256)
    print("Payload tensor shape (single sample):", payload_tensor.shape)  # Should be (256,)
    
    # Simulate batch dimension: (batch_size, max_payload_len)
    batch_payload = payload_tensor.unsqueeze(0)  # (1, 256)
    print("Batch payload shape:", batch_payload.shape)  # Should be (1, 256)
    
    # Extract first byte as ClassifyPayloadEncoder expects
    payload_head = batch_payload[:, 0]  # (1,)
    print("Payload head shape (first byte):", payload_head.shape)  # Should be (1,)
    print("Payload head value:", payload_head.item())  # Should be the first byte value (0x47 = 71)
    
    # Test ClassifyPayloadEncoder
    encoder = ClassifyPayloadEncoder(llm_dim=512)
    token = encoder(payload_head)
    print("Payload token shape:", token.shape)  # Should be (1, 512)
    print("✓ Dimension check passed: ClassifyPayloadEncoder works correctly with extracted first byte")