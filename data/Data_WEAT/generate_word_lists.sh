#!/bin/bash

python src/get_words_for_weat_weaft_tests.py --weat_wefat_file data/Data_WEAT/weat_attrib_target.json --output_file data/Data_WEAT/all_weat_words.txt

python src/get_words_for_weat_weaft_tests.py --weat_wefat_file data/Data_WEAT/weat_attrib_target_2.json --output_file data/Data_WEAT/all_weat2_words.txt

python src/get_words_for_weat_weaft_tests.py --weat_wefat_file data/Data_WEAT/weat_attrib_target_same_length.json --output_file data/Data_WEAT/all_weat_words_same_length.txt

python src/get_words_for_weat_weaft_tests.py --weat_wefat_file data/Data_WEAT/weat_attrib_target_2_same_length.json --output_file data/Data_WEAT/all_weat2_words_same_length.txt

python src/get_words_for_weat_weaft_tests.py --weat_wefat_file data/Data_WEAT/weat_attrib_target_same_length_lemma.json --output_file data/Data_WEAT/all_weat_words_same_length_lemma.txt

python src/get_words_for_weat_weaft_tests.py --weat_wefat_file data/Data_WEAT/weat_attrib_target_2_same_length_lemma.json --output_file data/Data_WEAT/all_weat2_words_same_length_lemma.txt