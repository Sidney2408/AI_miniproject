import argparse
import pickle
import time
from collections import Counter

import nltk

import mappings
from vqaTools.vqa import VQA



def build_vocab(annotation_file, question_file, threshold):
    """Build a simple vocabulary wrapper."""
    vqa = VQA(annotation_file, question_file)
    counter = Counter()
    start = time.time()
    
    # Iterate through all questions and count frequency of words.
    all_questions = vqa.questions["questions"]
    for i, question in enumerate(all_questions, 1):
        question = question["question"]
        tokens = nltk.tokenize.word_tokenize(question.lower())
        counter.update(tokens)
        
        if i % 1000 == 0:
            print("[{}/{}] Tokenized the captions. ({:.3f}s)".format(
                      i, len(all_questions), time.time()-start),
                  end="\r"
                 )
    
    # Final progress counter line.
    print("[{}/{}] Tokenized the captions. ({:.3f}s)".format(
                      i, len(all_questions), time.time()-start),
         )
    print("Using threshold: {}".format(threshold))
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    # Create a vocab wrapper and add some special tokens.
    vocab = mappings.Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(annotation_file=args.annotation_path,
                        question_file=args.question_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, 
                        default='annotations/v2_mscoco_train2014_annotations.json',
                        help='path for train annotation file')
    parser.add_argument('--question_path', type=str, 
                        default='questions/v2_OpenEnded_mscoco_train2014_questions.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=3,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)