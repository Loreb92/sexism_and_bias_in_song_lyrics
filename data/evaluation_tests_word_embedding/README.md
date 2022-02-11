This folder contains all the tests used to evaluate the quality of word embeddings downloaded from https://github.com/licstar/compare .

The file all_embedding_eval_words.txt contains all the unique words involved in the tests. It was obtained with the following command:

`python src/get_words_for_embedding_eval_tests.py --eval_files_path data/evaluation_tests_word_embedding/ --output_file data/evaluation_tests_word_embedding/all_embedding_eval_words.txt`