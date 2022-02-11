# take one embedding file, collect the words of interest from /evaluation/all_tests_words and save the new embedding file
# run all the tests


import re, glob, os, argparse, zipfile

from logger_script import init_logger, logger

from ws import func as eval_ws
from toefl import func as eval_tfl
from king import func as eval_king


def read_embedding_file_by_rows(file):
    
    if file.endswith('.txt'):
        
        with open(file, 'rt') as rr:
            for line in rr:
                yield line
        
    elif file.endswith('.zip'):

        with zipfile.ZipFile(file, 'r') as z:
            file_ = file.replace(".zip", ".txt").split("/")[-1]
            with z.open(file_, 'r') as f:
                for line in f:
                    line = line.decode("utf-8")
                    yield line
                    
    else:
        raise Exception("Error, can not handle this kind of file.")
                    

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='A test')
    parser.add_argument("--emb_dim", type=int, help="The dimensionality of the embedding")
    parser.add_argument("--eval_files_path", type=str, help="Folder where all the test words are stored")
    parser.add_argument("--embedding_file", type=str, help="File containing all the embeddings. It is assumed the format with one word vector per line like 'words dim1 dim2 ...'")
    parser.add_argument("--embedding_name", type=str, help="Name of the embedding")
    parser.add_argument("--results_dir", type=str, help="Directory to save results")
    parser.add_argument("--upper_and_lower", type=bool, default=False, help="Whether to keep both upper and lower case of each word")
    parser.add_argument("--n_processes", type=str, default=2, help="Number of parallel processes")
    parser.add_argument("--log_file", type=str, help="Logger file")       
    
    args = parser.parse_args()
    
    eval_files_path = args.eval_files_path
    upper_and_lower = args.upper_and_lower
    embedding_file = args.embedding_file
    embedding_name = args.embedding_name
    results_dir = args.results_dir
    n_processes = args.n_processes
    emb_dim = args.emb_dim
    log_file = args.log_file
    
    init_logger(log_file)
    logger.info(f'Start computation quality word embedding tests. Results will be saved at {results_dir}')
    
    # collect the words of interest and save it in a temporary file
    # remember to write the first line (n_words, emb_dim)
    words_for_tests = []
    with open(eval_files_path+'all_embedding_eval_words.txt', 'rt') as rr:
        for line in rr:
            line = line.strip()
            if line!='':
                words_for_tests.append(line.strip())
    
    if upper_and_lower:
        words_for_tests_lower = [w.lower() for w in words_for_tests]
        words_for_tests.extend(words_for_tests_lower)
        
    words_for_tests = set(words_for_tests)
    n_words_test = len(words_for_tests)
    #print('Number of unique words for tests: ', n_words_test)
    logger.info('Number of unique words for tests: {}'.format(n_words_test))
    
    # get the embedding file and store the vectors
    # we need to save it because the following evaluation scripts works with the embedding file
    tmp_emb_file = eval_files_path+'temp_embs.txt'
    words_for_tests_in_emb_file = 0
    logger.info(f'Reading word vectors...')
    with open(tmp_emb_file, 'wt') as ww:
        for n, line in enumerate(read_embedding_file_by_rows(embedding_file), 1):
            
            # check if the first line is a vector or vocab_len vec_dim
            if len(line.strip().split())==2:
                continue
            
            if n%10000==0:
                #print(f"Number of word vectors read: {n}...")
                logger.info(f"Number of word vectors read: {n}...")
            vec = line.strip().split()
            word, vec = vec[0], vec[1:]
            
            if word in words_for_tests:
                ww.write(line)
                words_for_tests_in_emb_file += 1
        
    #print(f'Number of words found: {words_for_tests_in_emb_file}/{n_words_test}')     
    logger.info(f'Number of word vectors found: {words_for_tests_in_emb_file}')
    
    # add the first line
    cmd = fr"sed -i '' '1s/^/{words_for_tests_in_emb_file} {emb_dim}\'$'\n/' {tmp_emb_file}"
    #print(cmd)
    logger.info(f'Running: {cmd}')
    os.system(cmd)

    # run the evals from folder /evaluations
    '''
    cmd_ws = f"python {eval_files_path}ws/ws.py --embedding_name {embedding_name} --embedding_file {tmp_emb_file} --results_dir {results_dir} --n_processes {n_processes} --eval_files_path {eval_files_path}" 
    #print(cmd_ws)
    logger.info(f'Running: {cmd_ws}')
    os.system(cmd_ws)
    '''
    logger.info(f'Running WS test...')
    if not os.path.exists(results_dir+embedding_name):
        os.makedirs(results_dir+embedding_name)
    eval_ws(eval_files_path+'ws',
           eval_files_path.rstrip("/"),
           'temp_embs.txt',
           results_dir+embedding_name,
           'ws_test.txt')
    
    '''
    cmd_tfl = f"python {eval_files_path}tfl/toefl.py --embedding_name {embedding_name} --embedding_file {tmp_emb_file} --results_dir {results_dir} --n_processes {n_processes} --eval_files_path {eval_files_path}" 
    #print(cmd_tfl)
    logger.info(f'Running: {cmd_tfl}')
    os.system(cmd_tfl)
    '''
    logger.info(f'Running TOEFL test...')
    eval_tfl(eval_files_path+'tfl',
           eval_files_path.rstrip("/"),
           'temp_embs.txt',
           results_dir+embedding_name,
           'toefl_test.txt')
    
    '''
    cmd_syn_sem = f"python {eval_files_path}syn_sem/king.py --embedding_name {embedding_name} --embedding_file {tmp_emb_file} --results_dir {results_dir} --n_processes {n_processes} --eval_files_path {eval_files_path}" 
    #print(cmd_syn_sem)
    logger.info(f'Running: {cmd_syn_sem}')
    os.system(cmd_syn_sem)
    '''
    logger.info(f'Running SYN-SEM test...')
    eval_king(eval_files_path+'syn_sem',
           eval_files_path.rstrip("/"),
           'temp_embs.txt',
           results_dir+embedding_name,
           'syn-sem_test.txt')
    
    # delete embedding of words of interest
    os.remove(tmp_emb_file)
    
    
    
    
    