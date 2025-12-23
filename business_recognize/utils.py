import binascii
import pandas as pd
import numpy as np
import torch
import zlib
import struct
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, LlamaModel, LlamaConfig, Qwen3Config, Qwen3Model

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

def load_plm_qwen3(model_path):
    model_config = Qwen3Config.from_pretrained(model_path)
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen3Model.from_pretrained(model_path, config=model_config)
    return model, tokenizer, model_config

# --- 1. IP 处理: Hex String -> [Int, Int, Int, Int] ---
def hex_ip_to_tensor(hex_ip_str):
    """
    例如: '0a010a01' -> torch.tensor([10, 1, 10, 1])
    """
    try:
        # 每2个字符切分并转为int
        octets = [int(hex_ip_str[i:i+2], 16) for i in range(0, len(hex_ip_str), 2)]
        if len(octets) != 4:
            return torch.zeros(4, dtype=torch.long)
        return torch.tensor(octets, dtype=torch.long)
    except:
        return torch.zeros(4, dtype=torch.long)

# --- 2. Payload 处理 (复用之前逻辑) ---
def preprocess_payload(hex_payload, max_len=256):
    s = str(hex_payload).strip()
    try:
        if len(s) % 2 != 0: s = s + '0'
        byte_indices = list(binascii.unhexlify(s))
    except:
        byte_indices = []

    current_len = len(byte_indices)
    if current_len > max_len:
        indices = byte_indices[:max_len]
    else:
        # Padding with 0
        indices = byte_indices + [0] * (max_len - current_len)
        
    return torch.tensor(indices, dtype=torch.long)

# --- 3. Host 处理 (复用之前逻辑) ---
def preprocess_host(host_str, max_len=64):
    # 处理 NaN 或 非字符串情况
    if pd.isna(host_str):
        host_str = ""
    
    s = str(host_str).lower().strip()
    indices = [ord(c) for c in s if 0 <= ord(c) < 128]
    
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [0] * (max_len - len(indices))
        
    return torch.tensor(indices, dtype=torch.long)

class TrafficDataset(Dataset):
    def __init__(self, dataframe, label_encoder, max_payload_len=256, max_host_len=64, has_label=True):
        """
        :param dataframe: 已经切分好的 pd.DataFrame
        :param label_encoder: 已经 fit 过的 sklearn LabelEncoder
        :param has_label: 是否包含 label 列
        """
        self.df = dataframe.reset_index(drop=True)
        self.le = label_encoder
        self.max_payload_len = max_payload_len
        self.max_host_len = max_host_len
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 提取并转换特征
        
        # Payload
        payload_tensor = preprocess_payload(row['payload'], self.max_payload_len)
        
        # Host
        host_tensor = preprocess_host(row['host'], self.max_host_len)
        
        # IP (Hex str -> Tensor Long)
        src_ip_tensor = hex_ip_to_tensor(row['hex_src_ip'])
        dst_ip_tensor = hex_ip_to_tensor(row['hex_dst_ip'])
        
        # Port (Int -> Float Normalized)
        # 端口归一化到 0-1 之间，有助于 MLP 收敛
        src_port_norm = float(row['src_port']) / 65535.0
        dst_port_norm = float(row['dst_port']) / 65535.0
        
        src_port_tensor = torch.tensor([src_port_norm], dtype=torch.float32)
        dst_port_tensor = torch.tensor([dst_port_norm], dtype=torch.float32)
        
        result = {
            'payload': payload_tensor,
            'src_ip': src_ip_tensor,
            'dst_ip': dst_ip_tensor,
            'src_port': src_port_tensor,
            'dst_port': dst_port_tensor,
            'host': host_tensor,
        }
        
        # Label (only if has_label is True)
        if self.has_label:
            label_str = row['label']
            # 处理可能出现的未知标签 (在测试集中)
            try:
                label_idx = self.le.transform([label_str])[0]
            except:
                label_idx = 0 # 或者定义一个 unknown 类
            result['label'] = torch.tensor(label_idx, dtype=torch.long)
        else:
            # Return dummy label (won't be used)
            result['label'] = torch.tensor(0, dtype=torch.long)
        
        return result
    
def get_traffic_dataloaders(csv_path, batch_size=32, test_size=0.2):
    # 1. 读取数据
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # (可选) 简单清洗，去除 payload 为空的行
    df = df.dropna(subset=['payload'])
    
    # 2. 处理标签 (Label Encoding)
    # 必须在切分前对所有 Label 进行 Fit，确保训练集和测试集的映射一致
    le = LabelEncoder()
    df['label'] = df['label'].astype(str)
    # Filter out NaN values (which become 'nan' string after astype(str))
    valid_labels = df['label'][df['label'] != 'nan']
    le.fit(valid_labels)
    
    print(f"Total Classes: {len(le.classes_)}")
    # print(f"Classes: {le.classes_}")
    
    # 3. 拆分训练集和测试集 (Stratified Split 保证类别分布均衡)
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['label']
    )
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 4. 实例化 Dataset
    train_dataset = TrafficDataset(train_df, le)
    test_dataset = TrafficDataset(test_df, le)
    
    # 5. 创建 DataLoader
    # num_workers 可以根据你的 CPU 核心数调整，通常设为 4 或 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, le

def smart_decode(hex_string):
    raw_bytes = binascii.unhexlify(hex_string)
    # 尝试 Gzip/Zlib 解压
    try:
        # 16 + zlib.MAX_WBITS 用于自动识别 gzip 头部
        decompressed = zlib.decompress(raw_bytes, 16 + zlib.MAX_WBITS)
        return decompressed.decode('utf-8', errors='ignore') # 解压后转文本
    except Exception:
        pass

    # 如果不是压缩包，尝试直接解码，处理非 UTF-8 (如 GBK)
    try:
        return raw_bytes.decode('gbk') 
    except:
        return raw_bytes.decode('utf-8', errors='ignore')

class PayloadFeatureExtractor:
    def __init__(self):
        pass

    def _hex_to_bytes(self, hex_str):
        try:
            return binascii.unhexlify(hex_str)
        except binascii.Error:
            return None

    def _parse_tls_sni(self, data):
        """
        解析 TLS Client Hello 包提取 SNI (Server Name Indication)
        """
        try:
            # 1. TLS Record Header (5 bytes)
            # Content Type (1) + Version (2) + Length (2)
            if len(data) < 5 or data[0] != 0x16:
                return None
            
            # 2. Handshake Header (4 bytes)
            # Handshake Type (1) + Length (3)
            # Client Hello Type is 0x01
            if len(data) < 9 or data[5] != 0x01:
                return None

            # 游标: 跳过 Record Header(5) + Handshake Header(4)
            cursor = 9

            # 3. Client Hello Body
            # Version(2) + Random(32)
            cursor += 34
            if cursor >= len(data): return None

            # Session ID
            session_id_len = data[cursor]
            cursor += 1 + session_id_len
            if cursor >= len(data): return None

            # Cipher Suites
            cipher_suites_len = struct.unpack('!H', data[cursor:cursor+2])[0]
            cursor += 2 + cipher_suites_len
            if cursor >= len(data): return None

            # Compression Methods
            comp_methods_len = data[cursor]
            cursor += 1 + comp_methods_len
            if cursor >= len(data): return None

            # Extensions (这是我们需要的)
            if cursor + 2 > len(data): return None
            extensions_len = struct.unpack('!H', data[cursor:cursor+2])[0]
            cursor += 2
            
            end_of_extensions = cursor + extensions_len
            
            while cursor < end_of_extensions:
                if cursor + 4 > len(data): break
                ext_type = struct.unpack('!H', data[cursor:cursor+2])[0]
                ext_len = struct.unpack('!H', data[cursor+2:cursor+4])[0]
                cursor += 4
                
                # Extension Type 0x0000 is Server Name (SNI)
                if ext_type == 0x0000:
                    if cursor + 2 > len(data): break
                    sni_list_len = struct.unpack('!H', data[cursor:cursor+2])[0]
                    cursor += 2
                    
                    if cursor + 3 > len(data): break
                    # sni_type (1) + server_name_len (2)
                    server_name_len = struct.unpack('!H', data[cursor+1:cursor+3])[0]
                    cursor += 3
                    
                    if cursor + server_name_len > len(data): break
                    server_name = data[cursor:cursor+server_name_len]
                    return server_name.decode('utf-8')
                
                cursor += ext_len

            return None
        except Exception:
            return None

    def _parse_http_smart(self, data):
        """
        智能解析 HTTP，处理 Gzip 压缩
        """
        try:
            # 尝试直接解码 Header 部分
            # 查找 HTTP Header 和 Body 的分隔符 \r\n\r\n (0d0a0d0a)
            split_idx = data.find(b'\r\n\r\n')
            
            decoded_text = ""
            
            if split_idx != -1:
                # 分离 Header 和 Body
                header_bytes = data[:split_idx]
                body_bytes = data[split_idx+4:]
                
                # 解码 Header
                header_text = header_bytes.decode('utf-8', errors='ignore')
                decoded_text += header_text + "\n\n[Body Parsing]:\n"
                
                # 检查是否包含 Gzip 头 (1f 8b)
                if b'Content-Encoding: gzip' in header_bytes or (len(body_bytes) > 2 and body_bytes[:2] == b'\x1f\x8b'):
                    try:
                        # 尝试 Gzip 解压
                        decompressed_body = zlib.decompress(body_bytes, 16 + zlib.MAX_WBITS)
                        decoded_text += decompressed_body.decode('utf-8', errors='ignore')
                    except Exception:
                        decoded_text += f"[Gzip Decompression Failed] Raw Body Len: {len(body_bytes)}"
                else:
                    # 尝试普通解码
                    decoded_text += body_bytes.decode('utf-8', errors='ignore')
            else:
                # 没有找到 Header 结尾，尝试暴力解码
                decoded_text = data.decode('utf-8', errors='ignore')

            return decoded_text.strip()
        except Exception:
            return None

    def extract(self, hex_payload):
        """
        主入口函数
        """
        if not hex_payload or str(hex_payload) == 'nan':
            return "Empty Payload"

        data = self._hex_to_bytes(hex_payload)
        if not data:
            return "Invalid Hex"

        # 策略分发
        # 16 = TLS Handshake
        if data[0] == 0x16:
            sni = self._parse_tls_sni(data)
            if sni:
                return f"[HTTPS/TLS] SNI: {sni}"
            else:
                return "[HTTPS/TLS] Encrypted Handshake (No SNI found)"
        
        # 47 = 'G' (GET), 50 = 'P' (POST/PUT), 48 = 'H' (HTTP), 44 = 'D' (DELETE)
        elif data[0] in [0x47, 0x50, 0x48, 0x44]:
            content = self._parse_http_smart(data)
            # 截取前 200 个字符避免输出太长
            preview = content[:200].replace('\n', ' ') + "..." if len(content) > 200 else content
            return f"[HTTP] Content: {preview}"
            
        else:
            return f"[Unknown Protocol] Header Hex: {binascii.hexlify(data[:4]).decode()}"

def count_payload_prefix_combinations(csv_file_path):
    """
    统计 HTTP.csv 中 payload 字段的前两个十六进制字符有多少种不同的组合。
    
    Args:
        csv_file_path: HTTP.csv 文件的路径
        
    Returns:
        dict: 包含统计结果的字典
            - 'total_count': 总记录数
            - 'unique_prefixes': 不同前缀组合的数量
            - 'prefix_counts': 每个前缀组合出现的次数（字典）
            - 'prefix_list': 所有唯一前缀的列表（排序后）
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(csv_file_path)
        
        # 检查是否有 payload 列
        if 'payload' not in df.columns:
            print(f"错误: CSV 文件中没有找到 'payload' 列")
            return None
        
        # 提取 payload 列，去除空值
        payloads = df['payload'].dropna()
        
        # 统计前两个字符的组合
        prefix_set = set()
        prefix_counts = {}
        
        for payload in payloads:
            # 转换为字符串并去除空格
            payload_str = str(payload).strip()
            
            # 如果长度小于2，跳过
            if len(payload_str) < 2:
                continue
            
            # 提取前两个字符（转换为小写以便统一）
            prefix = payload_str[:2].lower()
            
            # 验证是否为有效的十六进制字符
            if all(c in '0123456789abcdef' for c in prefix):
                prefix_set.add(prefix)
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        # 排序前缀列表
        prefix_list = sorted(prefix_set)
        
        result = {
            'total_count': len(payloads),
            'unique_prefixes': len(prefix_set),
            'prefix_counts': prefix_counts,
            'prefix_list': prefix_list
        }
        
        return result
        
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_file_path}' 未找到")
        return None
    except Exception as e:
        print(f"错误: 处理文件时发生异常: {str(e)}")
        return None

if __name__ == "__main__":

    '''DataLoader 测试代码'''
    # # 假设文件名为 HTTP.csv
    # csv_file = 'data/HTTP.csv' # 请替换为实际路径
    
    # # 这里的逻辑仅为演示，实际运行时请确保 csv 文件存在
    # train_loader, test_loader, label_encoder = get_traffic_dataloaders(csv_file)
    
    # # 模拟一个 Batch 验证数据格式
    # for batch in train_loader:
    #     print("Payload shape:", batch['payload'].shape)   # (32, 256)
    #     print("Src IP shape:", batch['src_ip'].shape)     # (32, 4)
    #     print("Src Port shape:", batch['src_port'].shape) # (32, 1)
    #     print("Label shape:", batch['label'].shape)       # (32,)
    #     break

    '''PLM 加载测试代码'''
    # model_path = '../downloaded_plms/qwen/large'
    # model, tokenizer, config = load_plm_qwen3(model_path)
    # print(model)

    '''Payload 特征提取测试代码'''
    # extractor = PayloadFeatureExtractor()
    # sample_data = [
    #     # HTTP (GET)
    #     "474554202f6c6f672f643d483473494141414141414141414b5653793237624d42443846594e6e53785a46697052384339496b645a733661654f6d4c57424134474e6c45355a45515325324645616672764a57576e6a304e50765167374f395475374f353852373270414330786f31464743474d595a2532426b636c586233414c58652532464d58786d4d665248496d6d7966746a34776a4530527756706467526a5a614f4b5778626966344d4e4f786167484d344767585864707951366236597771426c49636f4f504c7731492532467742723666667a7641415239656e4b63555232687847714876585569686c427863745563776f78314643434535597a44684c47585638627739514f785959797a6a4a424f5552533557696f436b555747424b34714c67306a38646a5161374f55305476654c637542485176752532426235586178585978744264714966584c6f6a6d462532466245486f554e6c71753769353255445862786478464f4f4978576c794531624e4a4742765776336f5336313870586f6f79376e2532464f4171652532426c5664574a6531395150413462363143726f4f744f4d364c35706c42546a68555942314c494d306a576a67354f4e413067673470544c6d484e4472546b394b5931495569727635694a417834545352557159704238307a49744d7346756830745248617a6c6a664249636b784b4572797878313275376a4c314c557572564735253246426b6325324625324251304e424e474745337058733944464e506e55565a6c696f536941776e416558434b565361424a6c4d704d42456b5568356439674757744862646932386b39446c6838744c6e253242317552623062784d376e6e766454356e64253246374138685779646a32703030586e396c4e5a514f7237253242794a48766e45716f5a76465a334779464e4d484c78636f356650673669645065705a687451253242396f364c78766f35724e567257627637395958614c4c33253242655958622532464c3774336672712532467a25324239754c623153652532466a62327459543155253246335258615958253242334a6225324634354166507747456253584764774d41414125334425334426733d623338393233353864386231623861613365363862366462323930353565646620485454502f312e310d0a436f6e6e656374696f6e3a204b6565702d416c6976650d0a557365722d4167656e743a2044616c76696b2f322e312e3020284c696e75783b20553b20416e64726f69642031303b204e583635394a204275696c642f514b51312e3230303430352e303032290d0a486f73743a2076726c6f672e7479726561642e636f6d0d0a4163636570742d456e636f64696e673a20677a69700d0a0d0a485454502f312e3120323030204f4b0d0a446174653a204672692c2033312044656320323032312030363a35343a333120474d540d0a436f6e74656e742d547970653a20746578742f68746d6c0d0a436f6e74656e742d4c656e6774683a20310d0a436f6e6e656374696f6e3a206b6565702d616c6976650d0a4163636573732d436f6e74726f6c2d416c6c6f772d4f726967696e3a202a0d0a4163636573732d436f6e74726f6c2d416c6c6f772d486561646572733a202a0d0a5365727665723a20656c620d0a0d0a31",
    #     # HTTPS (Ximalaya)
    #     "1603010200010001fc0303d4e638a5f395ea381738bdfb46e73faab0509e6538162a3cca5dc4133ecb1a4c20983bbf31c0e701f8d604893cf6b947083d70b85e3dad149185795ca6278f2d94001e130113021303c02bc02ccca9c02fc030cca8c013c014009c009d002f0035010001950000001900170000146d65726d6169642e78696d616c6179612e636f6d00170000ff01000100000a00080006001d00170018000b00020100002300d014c581ee4a31c32c32fc69e208772b8eb88af409336381001dae02658e550aa84055ee587471d632bee455e76f560dd3657103723a64a848e29666d7736ae1a370d601723ba7a3c6e3b3900436c0919b87e1855e0e38fbd4ca94f9c634708386a78ed8beb44be1ac6bb87b5b141430fe41e95d43e9d1860734c48ac33fcfdb443da4424110bde737cad224fc79e50757d7e57fd0c0a817d8a7c80060e1ff8d1e7b4034dbfe9ad51d2dc66d9eb51af6f2213ea9347071e983ab2f0d0058c3732b90077e8f7127b1f441f4571bd7d92c930010000e000c02683208687474702f312e31000500050100000000000d00140012040308040401050308050501080606010201003300260024001d0020b80a2153eafdcc2a4722937e61795b18ea773b0b6154373efd3c0c0774f38679002d00020101002b000908030403030302030100150015000000000000000000000000000000000000000000160303005e0200005a0303129eb191bae7f2d6402fdf2b65744466fdbe1ba10a773e6047858d4e52b3cb7420983bbf31c0e701f8d604893cf6b947083d70b85e3dad149185795ca6278f2d94c030000012ff0100010000100005000302683200170000140303000101160303002861a50cc196109849b3497d02d30367c950a9c3fda8230349605f5afd26e39d1b35837a0ba93e638014030300010116030300280000000000000000a2a3a9c0509768e67e5b2729a4bbff9d6c48f2fa696545d0ba9644c6d05235ba170303004061a50cc19610984a2a6e3876849b803b42102446d2c627adaeea79d1b727f974d4580492bf8d582ece3b2eaaa1d2ef2e73b983717542583885f6c6bad1ce0bef170303003000000000000000017c625599d64139e7f06b3653901d8502ac2f173a30b9d8a1ced931c5b55a2b5f98483b426573f32b1703030027000000000000000206c71b5b8e67cb46229938bce8de548e3091ddfc70c377c740f5386c6277bb170303002500000000000000032ff563617576f96fd538c7e3eb9eea9a347b90f9228233c549f93907eb17030303660000000000000004b817166f7efab78962cc2c789d5bcb8da23e7e1126a549444c8ee58fe0936414a90cc7dd5a35c404c3be33e76f67b7b4a6870a578fb9ed6cc4cd532fa45e12c03c9a319948390b5a0ecadd83636909e93fa2cffc8df6be6a56fef4131a1eca82aab32cc1b4d234f15c081a1369c845e95db60433b9d48cc9758062553dce879a254de0e564d62deb5ffaee93b0957f91b36a0858f74371b5e29710e2df18b2ee99cf97458d3a5c350be139a5109ab9abc0f32923c08427838914f0f2ff646e0a1c27ba72b732151952abf7f6c7b6f8ce95fa094eadade8ef3872e3d48d833e8ec78409882e06a0a69d95db963cce32dd7c9e3e5a354cf0c92548815cc8e3b67d89cbf7afa5804bcbf12ee31da88c4efb156b0b79f6ad44492b5381234fdb4ac15e25238b71a9c5337a2ba1eaf2daee1b940964cb840006afd32c873ff56209b693585d524bba7193f641919ca9e03ff7346f792e2f6ae65c6f0edf57894440695acd81713a3039fb1a48f561f449564a0b87c68611a7fe38c43ea6326924d59a768de66adfde139170028a8000978c69808e6566ad4fd2caff27683fdacc8f388c92f00c4c783820a2368b52cc1bdaf21017250a9a5baa531284ffb8ce9b75dacaf59309bbec30043f2914478dab42b95166133ed855b7c1b49b7e019c28b60d69db524e76778b8b6338b17b8388e69a8574fb23128f3bd75ea980b943a456fa50236dcdaf145dcc5d9f5626dcb7dc0fcc1d103e2d09998a0e97cde159237ad456555d573e118ada7ba22ebe49cdef9f73",
    #     # HTTP (POST)
    #     "1603010200010001fc03030a1fc3e28e68b2fd1f0246edba4687ae48d0b9c4904f650998967f0c4b48f67c20b1b8b6b2e8143c4ae039b2e6768b1d2b126379698810a39ae8c45fa42b5564d2001e130113021303c02bc02ccca9c02fc030cca8c013c014009c009d002f0035010001950000001800160000136d6f62696c652e78696d616c6179612e636f6d00170000ff01000100000a00080006001d00170018000b00020100002300000010000e000c02683208687474702f312e31000500050100000000000d00140012040308040401050308050501080606010201003300260024001d0020cc706577f19dc1004ca3f0924079f3c64a2c1039721cb1e6ce2ca3b7f2a99471002d00020101002b0009080304030303020301001500e60000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000160303004e0200004a0303bb80ba0c8bb02b4a50f5606f4e930b057d39f6914a09578b68ed9ac7ada605e400c02f000022ff0100010000000000000b000403000102002300000010000500030268320017000016030312d80b0012d40012d10006a8308206a43082058ca003020102020c0e239f3fb593b4e7d59ccf48300d06092a864886f70d01010b05003050310b300906035504061302424531193017060355040a1310476c6f62616c5369676e206e762d7361312630240603550403131d476c6f62616c5369676e20525341204f562053534c2043412032303138301e170d3231313231363032353130375a170d3233303131373032353130375a3077310b300906035504061302434e310f300d06035504080c06e4b88ae6b5b7310f300d06035504070c06e4b88ae6b5b7312d302b060355040a0c24e4b88ae6b5b7e5969ce9a9ace68b89e99b85e7a791e68a80e69c89e99990e585ace58fb83117301506035504030c0e2a2e78696d616c6179612e636f6d30820122300d06092a864886f70d01010105000382010f003082010a0282010100afd787acbfc3f42a034408dbf8a484848adf78711f128e6de855f90b8301d91c4fe4292c4465cf847d0a80368a4db18bb36cc57c3c5b89b8a2084c4c91fd082d7d3681bb9b7e3a90b8bf376d7c23e34b80a2459537155c53b151567e9755d71214280bca98af8930024c3b21247c7cb3b82ad8c8a18cf558dff33e782d8bb9a0751f783633bf02bb7a3a73ae2054be59a41e3a225a7a6184fb826ee8afe9d02708fad9334b1a972843c3460a4ce08100922d01227bf992ba5ce935c5783b427766436916320b963c8da8f632a00231cae32cd54162f496be4e69d2cca817002563bb45dea7f49fc7ff340cbe649d4b3c6a742d0d83e04333749722efba1a816b0203010001a382035530820351300e0603551d0f0101ff0404030205a030818e06082b06010505070101048181307f304406082b060105050730028638687474703a2f2f7365637572652e676c6f62616c7369676e2e636f6d2f6361636572742f67737273616f7673736c6361323031382e637274303706082b06010505073001862b687474703a2f2f6f6373702e676c6f62616c7369676e2e636f6d2f67737273616f7673736c63613230313830560603551d20044f304d304106092b06010401a03201143034303206082b06010505070201162668747470733a2f2f7777772e676c6f62616c7369676e2e636f6d2f7265706f7369746f72792f3008060667810c01020230090603551d1304023000303f0603551d1f043830363034a032a030862e687474703a2f2f63726c2e676c6f62616c7369676e2e636f6d2f67",
    # ]
    # print("--- 逐条测试结果 ---")
    # for raw_hex in sample_data:
    #     #print(extractor.extract(raw_hex))
    #     print(smart_decode(raw_hex))
    
    '''统计功能测试代码'''
    # csv_file = 'data/HTTP.csv'
    # result = count_payload_prefix_combinations(csv_file)
    
    # if result:
    #     print(f"总记录数: {result['total_count']}")
    #     print(f"不同前缀组合数量: {result['unique_prefixes']}")
    #     print(f"\n前10个最常见的前缀组合:")
    #     sorted_prefixes = sorted(result['prefix_counts'].items(), key=lambda x: x[1], reverse=True)
    #     for prefix, count in sorted_prefixes[:10]:
    #         print(f"  {prefix}: {count} 次")
    #     print(f"\n所有唯一前缀列表 (共 {len(result['prefix_list'])} 个):")
    #     print(result['prefix_list'])
    
    