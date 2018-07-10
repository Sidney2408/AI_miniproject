import argparse
import pickle
import time
from collections import Counter

import mappings
from vqaTools.vqa import VQA



def build_ans(annotation_file, question_file, number):
    """Build a simple answer wrapper."""
    vqa = VQA(annotation_file, question_file)
    counter = Counter()
    start = time.time()
    
    # Iterate through all questions and count frequency of words.
    all_annotations = vqa.dataset["annotations"]
    for i, annotation in enumerate(all_annotations, 1):
        answers = annotation["answers"]
        answers = [a["answer"].lower() for a in answers]
        counter.update(answers)
        
        if i % 1000 == 0:
            print("[{}/{}] Counting the answers. ({:.3f}s)".format(
                      i, len(all_annotations), time.time()-start),
                  end="\r"
                 )
    
    # Final progress counter line.
    print("[{}/{}] Tokenized the captions. ({:.3f}s)".format(
                      i, len(all_annotations), time.time()-start),
         )
    
    # If the word frequency is less than 'threshold', then the word is discarded.
    top_ans = [ans for ans,freq in counter.most_common(number)]
    
    # Create a answer wrapper and add don't know answer.
    answers = mappings.Answer()
    answers.add_ans("<don't know>")
    
    # Add the answers to the wrapper.
    for i, a in enumerate(top_ans):
        answers.add_ans(a)
    return answers

def main(args):
    answers = build_ans(annotation_file=args.annotation_path,
                    question_file=args.question_path,
                    number=args.number)
    ans_path = args.ans_path
    with open(ans_path, 'wb') as f:
        pickle.dump(answers, f)
    print("Total number of answers: {}".format(len(answers)))
    print("Saved the answer wrapper to '{}'".format(ans_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, 
                        default='annotations/v2_mscoco_train2014_annotations.json',
                        help='path for train annotation file')
    parser.add_argument('--question_path', type=str, 
                        default='questions/v2_OpenEnded_mscoco_train2014_questions.json',
                        help='path for train annotation file')
    parser.add_argument('--ans_path', type=str, default='answers.pkl',
                        help='path for saving answer wrapper')
    parser.add_argument('--number', type=int, default=3000,
                        help="number of answers to use. (will have n+1 answers, 0th answer is don't know)")
    args = parser.parse_args()
    main(args)