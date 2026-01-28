from transformers import AutoTokenizer
from collections import defaultdict


def wordpiece(training_corpus, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    word_freqs = defaultdict(int)
    for text in training_corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")

    alphabet.sort()

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

    # Do NOT add your above this line.
    #======
    
    # Add your code here.
    tokenized_words = {}
    for word in word_freqs:
        # 首字符原样，其余字符带 ## 前缀
        chars = [word[0]] + [f"##{c}" for c in word[1:]]
        tokenized_words[word] = chars

    # 2. 不断做 BPE-style 的 pair 合并直到达到目标大小
    while len(vocab) < vocab_size:
        # 2.1 统计所有相邻 token 对出现的加权频次
        pair_counts = defaultdict(int)
        for word, tokens in tokenized_words.items():
            freq = word_freqs[word]
            for a, b in zip(tokens, tokens[1:]):
                pair_counts[(a, b)] += freq

        if not pair_counts:          # 如果没有可继续合并的 pair
            break

        # 2.2 找到出现频次最高的 token 对
        best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]

        # 2.3 生成合并后的 token，规则：去掉右 token 的前导 ## 再拼接
        left, right = best_pair
        merged_token = left + right.lstrip("#")  # remove leading "##" from right

        # 如果新 token 已在词表中，说明提前收敛
        if merged_token in vocab:
            break
        vocab.append(merged_token)

        # 2.4 更新所有单词的 token 序列
        for word, tokens in tokenized_words.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 如果当前和下一位正好是 best_pair，就替换成 merged_token
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokenized_words[word] = new_tokens

    #======
    # Do NOT add your below this line.

    return vocab

if __name__ == "__main__":
    default_training_corpus = [
        "peking university is located in haidian district",
        "computer science is the flagship major of peking university",
        "the school of electronic engineering and computer science enrolls approximately five hundred new students each year"  
    ]

    default_vocab_size = 120

    my_vocab = wordpiece(default_training_corpus, default_vocab_size)

    print('The vocab:', my_vocab)
    print('Vocab size:', len(my_vocab))

    def encode_word(custom_vocab, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in custom_vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(custom_vocab, text):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        encoded_words = [encode_word(custom_vocab, word) for word in pre_tokenized_text]
        return sum(encoded_words, [])

    print('Tokenization result:', tokenize(my_vocab, 'xylophone music is relaxing'))
