import multiprocessing
import os
import time
import argparse


def func(src_dir, vec_dir, vec_file_name, ret_dir, res_file_name):
    arg = './%s/compute-accuracy-txt %s/%s 0 < %s/questions-words.txt > %s/%s' % (src_dir, vec_dir, vec_file_name, src_dir, ret_dir, res_file_name)
    print('Calling command: ', arg)
    os.system(arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A test')
    parser.add_argument("--eval_files_path", type=str, help="Folder where all the tests are stored")
    parser.add_argument("--embedding_name", type=str, help="Name of the embedding under evaluation")
    parser.add_argument("--embedding_file", type=str, help="File containing all the embeddings. It is assumed the format with one word vector per line like 'words dim1 dim2 ...'")
    parser.add_argument("--results_dir", type=str, help="Name of the directory containing the results")
    parser.add_argument("--n_processes", type=int, default=2, help="Number of parallel processes.")
    args = parser.parse_args()
    
    n_processes = args.n_processes
    embedding_file = args.embedding_file
    embedding_name = args.embedding_name
    results_dir = args.results_dir
    eval_files_path = args.eval_files_path
    pool = multiprocessing.Pool(processes=n_processes)

    #vec_dir = "vec_%s" % model
    #ret_dir = "ret_king_%s" % model
    #vec_dir = "/Users/lorenzo/Projects/Bias_word_embeddings/WEAT_test/emb%s" % model
    #ret_dir = "ret_king_%s" % model
    #vec_dir = embedding_dir
    res_dir = results_dir+"%s" % embedding_name
    src_dir = eval_files_path+'syn_sem'

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
        
    vec_dir = "/".join(embedding_file.split("/")[:-1])
    vec_file_name = embedding_file.split("/")[-1]
    res_file_name = 'syn-sem_test.txt'
    if not os.path.exists(os.path.join(res_dir, res_file_name)):
        
        print(f'Evaluating embedding {embedding_name}, task syn-sem...')
        pool.apply_async(func, (src_dir, vec_dir, vec_file_name, res_dir, res_file_name, ))
        
    pool.close()
    pool.join()
    print("Sub-process(es) syn_sem tests done.")
    
    
