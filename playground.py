from minbpe import RegexTokenizer, BasicTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
tokenizer = RegexTokenizer(GPT4_SPLIT_PATTERN)
# tokenizer = BasicTokenizer()

text = "aaabdaaabac"
tokenizer.train(text, 256+3, False)
# tokenizer.merges, tokenizer.vocab

print(tokenizer.encode(text, 'none'))