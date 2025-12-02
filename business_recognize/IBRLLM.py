import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
import binascii
from urllib import parse
from transformers import AutoTokenizer, LlamaModel, LlamaConfig

def load_plm_llama(model_path):
    pad_token = '[PAD]'

    model_config = LlamaConfig.from_pretrained(model_path)
    #model_config.num_hidden_layers = 32
    model_config.output_hidden_states = True
    model_config.output_attentions = True

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': pad_token})
    tokenizer.pad_token = pad_token

    model = LlamaModel.from_pretrained(model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, model_config

def preprocess_payload(hex_payload, max_len=256):
    s = str(hex_payload).strip().replace(' ', '').replace('\n', '').replace('\r', '')
    try:
        # 如果长度是奇数，binascii 会报错，通常在前面补 0 修复
        if len(s) % 2 != 0:
            s = '0' + s
        byte_indices = list(binascii.unhexlify(s))
        
    except (binascii.Error, ValueError):
        # 如果遇到非 Hex 字符等异常情况，返回全 0 或特定的 Token
        byte_indices = []

    # Truncate or Pad
    current_len = len(byte_indices)
    
    if current_len > max_len:
        indices = byte_indices[:max_len]
    else:
        # pad 0x00
        padding_len = max_len - current_len
        indices = byte_indices + [0] * padding_len
    return torch.tensor(indices, dtype=torch.long)

def preprocess_host(host_str, max_len=64):
    # 1. 统一小写
    s = str(host_str).lower().strip()
    
    # 2. 转换为 ASCII 码列表
    # 使用 ord(c) 获取 ASCII，只保留 0-127 之间的字符
    indices = [ord(c) for c in s if 0 <= ord(c) < 128]
    
    # 3. 截断或填充
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [0] * (max_len - len(indices)) # 0 作为 padding
        
    return torch.tensor(indices, dtype=torch.long)

class SocketEncoder(nn.Module):
    def __init__(self, llm_dim, hidden_dim=64):
        super(SocketEncoder, self).__init__()
        self.ip_embedding = nn.Embedding(256, hidden_dim)
        self.port_embedding = nn.Sequential(
            nn.Embedding(1, hidden_dim),
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
        self.key_dim = key_dim
        self.llm_dim = llm_dim
        
        self.q_proj = nn.Linear(input_dim, key_dim * num_heads)
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

#TODO: finish the IBRLLM class
class IBRLLM(nn.Module):
    def __init__(self, args):
        super(IBRLLM, self).__init__()
        self.args = args
        self.tiny_vocab_size = 1000
        self.llm_dim = args.llm_dim
        self.plm_embed_size = self.llm_dim
        self.payload_max_len = args.payload_max_len
        self.host_max_len = args.host_max_len

        self.plm, self.tokenizer, self.plm_config = load_plm_llama(args.model_path)
        self.word_embeddings = self.plm.get_input_embeddings().weight
        self.vocabulary_size = self.word_embeddings.shape[0]
        self.device = args.device

        self.socket_encoder = SocketEncoder(self.llm_dim)
        self.payload_encoder = PayloadEncoder(self.llm_dim, max_len=self.payload_max_len)
        self.host_encoder = HostEncoder(self.llm_dim, max_len=self.host_max_len)

        self.mapping_layer = nn.Linear(self.vocabulary_size, self.tiny_vocab_size)
    def forward(self, payload, src_ip, dst_ip, src_port, dst_port, host):
        src_socket_token, dst_socket_token = self.socket_encoder(src_ip, src_port, dst_ip, dst_port)
        payload_token = self.payload_encoder(payload)
        host_token = self.host_encoder(host)
        
        return src_socket_token, dst_socket_token, payload_token, host_token

if __name__ == "__main__":
    sample_hex = "1603010200010001fc03037452567fd2dae79ec2063e47284bdc9ce48b9727452c5ad5b8d826fe7b23ea50200b878db51a6c61d255331d30b8704c18463d8367dfa5de5184e5b9a18e029ed00022caca130113021303c02bc02fc02cc030cca9cca8c013c014009c009d002f0035000a010001916a6a0000000000190017000014696d677075622e636875616e676b69742e636f6d00170000ff01000100000a000a00084a4a001d00170018000b00020100002300000010000e000c02683208687474702f312e31000500050100000000000d00140012040308040401050308050501080606010201001200000033002b00294a4a000100001d0020a3bb495314b025d20f640571ab7f4ad58b0adbde37ae6fe5103353f8075ccc11002d00020101002b000b0adada0304030303020301001b00030200027a7a000100001500c400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000160303004e0200004a0303a391622912f8c1febdf00f8a685f85d4d324e67ff4dfed3ed919e47af4073ee200c02f000022ff0100010000000000000b00040300010200230000001000050003026832001700001603030bf50b000bf1000bee0006933082068f30820577a00302010202100148ed065112379b7e5739c802bfd665300d06092a864886f70d01010b05003059310b300906035504061302555331153013060355040a130c446967694365727420496e63313330310603550403132a526170696453534c20544c5320445620525341204d697865642053484132353620323032302043412d31301e170d3231313132313030303030305a170d3232313232313233353935395a301a3118301606035504030c0f2a2e636875616e676b69742e636f6d30820122300d06092a864886f70d01010105000382010f003082010a0282010100b5e6eee3df7e451863ac3892635f8a715da01de4d24cbb5b43f72e18f8f03837fcae28a94854833f03bc98e060174f423242c7bae808d9bcff47b3910ce60f05f11e8bc503dcd101168a2640c184d1cd5943a20c508cc5bba8f0005ca8135be5b384f0df7955f5ee1ee8ee8db99396862f73eed64eb4efadac1e38c17dcee798758f0e4fea32e6c1b4ad1f01654ea4cc68f638fd938ec44e6328692f29afdbcc7555804ecacbb62fb37cb49259f2c3d45ed9f6836a6d17c33f9902fe9cc8ab9b5e9f172b0907a830f1cdd8833afb38e3f41d380bc9fd967ed435020b0a7639e38425bc774b245ff54a7e4ba22221fb138252ce370bec9041e2cba74d681d65850203010001a38203903082038c301f0603551d23041830168014a48de5be7c79e470236d2e2934ad2358dcf5317f301d0603551d0e04160414f3ce6c951f94d02edbd5ef510b68bbee60f9bb6030290603551d1104223020820f2a2e636875616e676b69742e636f6d820d636875616e676b69742e636f6d300e0603551d0f0101ff0404030205a0301d0603551d250416301406082b0601050507030106082b0601050507030230819b0603551d1f0481933081903046a044a0428640687474703a2f2f63726c332e64696769636572742e636f6d2f526170696453534c544c5344565253414d697865645348413235363230323043412d312e63726c3046a044a0428640687474703a2f2f63726c342e64696769636572742e636f6d2f526170696453534c544c5344565253414d697865645348413235363230323043412d312e63726c303e0603551d20043730353033060667810c0102013029302706082b06010505070201161b687474703a2f2f7777772e64696769636572742e636f6d2f43505330818506082b06010505" 
    
    tensor_out = preprocess_payload(sample_hex, max_len=256)
    
    print(f"原始 Hex: {sample_hex}")
    print(f"转换后的 Tensor: {tensor_out}")
    print(f"Tensor 形状: {tensor_out.shape}")
    print(f"数据类型: {tensor_out.dtype}")