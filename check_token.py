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

# 推荐测试列表 (方案一的变体)
candidates = ["worst", "poor", "fair", "good", "great", "best"]
# model_path 换成你的 llama3.2 路径
check_tokens('./downloaded_plms/llama/large', candidates)