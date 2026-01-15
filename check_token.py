from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel
import torch
def check_tokens(model_path, words):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Testing words: {words}")
    for w in words:
        # 测试带空格和不带空格两种情况
        # 在生成任务中，通常模型会输出带前导空格的词（因为在 prompt 后面）
        w_with_space = " " + w
        
        ids = tokenizer.encode(w, add_special_tokens=False)
        ids_space = tokenizer.encode(w_with_space, add_special_tokens=False)
        
        print(f"Word: '{w}' -> IDs: {ids} (Length: {len(ids)})")
        print(f"Word: '{w_with_space}' -> IDs: {ids_space} (Length: {len(ids_space)})")
        
        if len(ids_space) == 1:
            print(f"✅ Recommended Anchor: '{w_with_space}' (ID: {ids_space[0]})")
        elif len(ids) == 1:
            print(f"⚠️ Warning: '{w}' is single token but '{w_with_space}' is not. Model might struggle.")
        else:
            print(f"❌ Avoid: '{w}' splits into multiple tokens.")
        print("-" * 20)

def load_plm_llama(model_path):
    pad_token = '[PAD]'

    model_config = AutoConfig.from_pretrained(model_path)
    #model_config.num_hidden_layers = 32
    model_config.output_hidden_states = True
    model_config.output_attentions = True

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': pad_token})
    tokenizer.pad_token = pad_token

    model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, model_config

model, tokenizer, model_config = load_plm_llama("./downloaded_plms/llama/base")
input = torch.randn(1, 10, model_config.hidden_size)
outputs = model(inputs_embeds=input)
# print(dir(model))
# print(model.lm_head)
print(outputs.hidden_states[-1].shape)  # 最后一层隐藏状态

# def load_plm_llama(model_path):
#     pad_token = '[PAD]'

#     model_config = LlamaConfig.from_pretrained(model_path)
#     #model_config.num_hidden_layers = 32
#     model_config.output_hidden_states = True
#     model_config.output_attentions = True

#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     tokenizer.add_special_tokens({'pad_token': pad_token})
#     tokenizer.pad_token = pad_token

#     model = LlamaModel.from_pretrained(model_path, config=model_config)
#     model.resize_token_embeddings(len(tokenizer))

#     return model, tokenizer, model_config

# model, tokenizer, model_config = load_plm_llama("./downloaded_plms/llama/base")
# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)
# print(dir(model))
# #print(model.lm_head)
# print(outputs.last_hidden_state.shape)  # 最后一层隐藏状态