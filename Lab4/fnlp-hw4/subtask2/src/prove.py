# -*- coding: utf-8 -*-
import argparse
import os
import json
from tqdm import tqdm

from dictionary import WordDictionary
from corpus import ParallelCorpus
from tokenizer import *
from prompt import *
from grammar import GrammarBook
from prompt import construct_prompt_za2zh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Linguistic resources
    parser.add_argument('--src_lang', type=str, default='za')
    parser.add_argument('--tgt_lang', type=str, default='zh')
    parser.add_argument('--dict_path', type=str, default='./data/dictionary_za2zh.jsonl')
    parser.add_argument('--corpus_path', type=str, default='./data/parallel_corpus.json')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.json')
    parser.add_argument('--grammar_book_path', type=str, default='./data/grammar_book.json')

    # Config for prompt
    parser.add_argument('--prompt_type', type=str, default='za2zh', help="Should be 'za2zh' for this project.")
    parser.add_argument('--num_parallel_sent', type=int, default=5,
                        help="Number of similar sentences to include in the prompt.")

    # Output path
    parser.add_argument('--output_path', type=str, default=None,
                        help="Path to save the generated prompt json file.")

    args = parser.parse_args()

    # Load linguistic resources
    print("Loading dictionary...")
    dictionary = WordDictionary(args.src_lang, args.tgt_lang, args.dict_path)
    print("Loading parallel corpus...")
    parallel_corpus = ParallelCorpus(args.src_lang, args.tgt_lang, args.corpus_path)
    print("Loading grammar book...")
    grammar_book = GrammarBook(args.grammar_book_path)
    print("Loading test data...")
    test_data = json.load(open(args.test_data_path, "r", encoding='utf-8'))

    # Prompt construction
    prompt_type_to_prompt_func = {
        'za2zh': construct_prompt_za2zh,
    }

    if args.prompt_type not in prompt_type_to_prompt_func:
        raise NotImplementedError("Unsupported prompt type!")
    else:
        prompt_func = prompt_type_to_prompt_func[args.prompt_type]

    # Setup output path
    if args.output_path is None:
        args.output_path = "./output/prompts_only_{}_parallel_{}.json".format(
            args.prompt_type, args.num_parallel_sent
        )
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("Generating prompts...")
    all_prompt_entries = []

    for item in tqdm(test_data, desc="Generating prompts"):
        src_sentence = item[args.src_lang]
        prompt = prompt_func(src_sentence, dictionary, parallel_corpus, grammar_book, args)

        log_entry = {
            "id": item['id'],
            "query": src_sentence,
            "gold": item[args.tgt_lang],
            "prompt": prompt,
            "source": item.get('source', 'unknown')
        }
        all_prompt_entries.append(log_entry)

    # Save to JSON
    with open(args.output_path, 'w', encoding='utf-8') as f_json:
        json.dump(all_prompt_entries, f_json, ensure_ascii=False, indent=2)

    print(f"Prompt JSON saved to {args.output_path}")