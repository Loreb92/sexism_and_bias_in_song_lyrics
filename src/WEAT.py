import numpy as np
from itertools import chain, combinations
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def cohend(d1, d2):
    ''' 
    Function to calculate Cohen's d for independent samples
    From https://machinelearningmastery.com/effect-size-measures-in-python/
    
    (https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)
    '''
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    
    return (u1 - u2) / s

def mean_cosine_similarity(X):
    '''
    Computes the average cosine similarity between the rows of X.
    
    Parameters:
    X : np.array, 2D array
    
    Returns:
    mean_sim : float, the average cosine similarity of the rows of X
    '''
    
    n_rows = X.shape[0]
    den = (n_rows**2 - n_rows) / 2
    
    mean_sim = np.triu(cosine_similarity(X), k=1).sum() / den
    return mean_sim


class BaseWordVecComparison():
    
    def _get_word_vectors(self, words):
        '''
        Assign the word vector to each word in the list words
        
        Parameters:
        words : list of str, the list of words to transform into vectors
        '''
        words_vector = np.vstack([self.word_vectors[w] for w in words if w in self.word_vecotrs_name])
        assert words_vector.shape[0]==len(words), "Some words have no embedding vector."
        
        return words_vector
    
    @staticmethod
    def _check_vector_set_pairs(arr_1, arr_2):
        ''' Just check whether the two vectors have same number of entries
        '''
        return arr_1.shape[0] == arr_2.shape[0]
    
    
    def s(self, w, arr_1=None, arr_2=None):
        '''
        Computes the association of the word vector w against the two attribute sets
        
        Parameters:
        w : np.array, 1D array representing the representation of the word w in an embedding space
        arr_1, arr_2 : np.array, 2D array where each row is a word embedding
        '''
        w_ = w[np.newaxis, ...]

        if arr_1 is None or arr_2 is None:
            cosine_with_A = cosine_similarity(w_, self.A_arr).flatten()
            cosine_with_B = cosine_similarity(w_, self.B_arr).flatten()
        else:
            cosine_with_A = cosine_similarity(w_, arr_1).flatten()
            cosine_with_B = cosine_similarity(w_, arr_2).flatten()

        s_ = cosine_with_A.mean() - cosine_with_B.mean()

        return s_
    


class WEAT(BaseWordVecComparison):
    '''
    Implements the WEAT and SC-WEAT.
    '''
    def __init__(self, A, B, word_vectors, set_names):
        '''
        Parameters:
        
        A, B : list of str, the list of attribute
        word_vectors : dict {str:np.array}, word vectors
        set_names : dict, it contains the name of each set
        '''
        
        self.A = A
        self.B = B
        self.set_names = set_names
        
        self.word_vectors = word_vectors
        self.word_vecotrs_name = set(word_vectors.keys())
        
        self.target_set_loaded = False
        self.target_sets_loaded = False
        
        # get the embeddings of each word in the sets
        self.A_arr = self._get_word_vectors(A)
        self.B_arr = self._get_word_vectors(B)
        self._check_vector_set_pairs(self.A_arr, self.B_arr)
             
       
    def add_target_set(self, W):
        '''
        Adds one single target set of words. Computes the association of each target word against the attribute sets
        
        Parameters:
        W : list of str, the target set
        '''
        self.W = W
        self.W_arr = self._get_word_vectors(W)
        self.target_set_loaded = True
        
        # compute associations with A and B
        self.WvsAB = [self.s(w) for w in self.W_arr]
        
        
    def add_target_sets(self, X, Y):
        '''
        Adds target sets of words X and Y. Computes the association of each target word against the attribute sets
        
        Parameters:
        X, Y : list of str, the target set
        '''
        
        self.X = X
        self.Y = Y
        self.X_arr = self._get_word_vectors(X)
        self.Y_arr = self._get_word_vectors(Y)
        self._check_vector_set_pairs(self.A_arr, self.B_arr)
        self.target_sets_loaded = True
        
        # compute associations with A and B
        self.XvsAB = [self.s(x) for x in self.X_arr]
        self.YvsAB = [self.s(y) for y in self.Y_arr]
        
   
    
    def compute_effect_size(self, approximate=True, n_iters=1000, compute_single_category_association=True):
        '''
        Computes the effect size and the p-value.
        
        Returns:
        effect_size : float, the effect size
        p : float, the p-value
        '''
        
        assert self.target_sets_loaded, "No target sets to compute the effect size."
        
        # get the measured test statistic
        test_statistic = sum(self.XvsAB) - sum(self.YvsAB)
        
        # get the p-value
        XvsAB_YvsAB = np.hstack([self.XvsAB, self.YvsAB])
        n_X, n_Y = len(self.XvsAB), len(self.YvsAB)
        
        if approximate:
            test_stats_rand = []
            for _ in range(n_iters):
                np.random.shuffle(XvsAB_YvsAB)
                XvsAB_new, YvsAB_new = XvsAB_YvsAB[:n_X], XvsAB_YvsAB[n_X:]
                
                test_statistic_rand = sum(XvsAB_new) - sum(YvsAB_new)
                test_stats_rand.append(test_statistic_rand)

            test_stats_rand = np.array(test_stats_rand)

        else:
            raise NotImplementedError("Exact computation should be implemented.")

        p = (test_stats_rand > test_statistic).sum() / test_stats_rand.shape[0]
        
        # compute the effect size
        effect_size = cohend(self.XvsAB, self.YvsAB)
        
        
        # compute single category association for X and Y
        single_category_associations = None
        if compute_single_category_association:
            single_category_associations = {}
            for W, W_name in zip([self.X, self.Y], ['X', 'Y']): 
                self.add_target_set(W)
                assoc, p_assoc = self.compute_single_category_association(approximate, n_iters)
            
                single_category_associations[W_name] = {'score':assoc, 'p':p_assoc}
        
        return effect_size, p, single_category_associations

    
    def compute_single_category_association(self, approximate=True, n_iters=1000):
        '''
        Computes the single category association test (SC-WEAT) as defined in "Gender Stereotypes in Natural Language: Word Embeddings Show Robust Consistency Across Child and Adult Language Corpora of More Than 65 Million Words"
        
        
        '''

        assert self.target_set_loaded, "No target set to compute the single category association test."
        
        # get observed test statistic
        test_statistic = np.mean(self.WvsAB)
        
        # get the p-value
        AB_arr = np.vstack([self.A_arr, self.B_arr])
        n_A, n_B = self.A_arr.shape[0], self.B_arr.shape[0]
        
        if approximate:
            test_stats_rand = []
            for _ in range(n_iters):
                np.random.shuffle(AB_arr)
                A_arr_new, B_arr_new = AB_arr[:n_A], AB_arr[n_A:]
                
                WvsAB_rand = [self.s(w, A_arr_new, B_arr_new) for w in self.W_arr]
                
                test_statistic_rand = np.mean(WvsAB_rand)
                test_stats_rand.append(test_statistic_rand)
            test_stats_rand = np.array(test_stats_rand)
            
        else:
            raise NotImplementedError("Exact computation should be implemented.")
            
        # compute left/right tale p-value depending on the sign of test_statistic
        if test_statistic>0:
            p = (test_stats_rand > test_statistic).sum() / test_stats_rand.shape[0]
        else:
            p = (test_stats_rand < test_statistic).sum() / test_stats_rand.shape[0]
        
        # compute the single cat association size
        effect_size = np.mean(self.WvsAB) / np.std(self.WvsAB, ddof=1)
        
        return effect_size, p
    
    
    def compute_effect_size_bootstrap(self, n_iters=1000):
        
        assert self.target_sets_loaded, "No target sets to compute the effect size."
        
        # get the measured test statistic
        test_statistic = sum(self.XvsAB) - sum(self.YvsAB)
        n_X, n_Y = len(self.XvsAB), len(self.YvsAB)
        
        effect_sizes_b = []
        for _ in range(n_iters):
            sample_x = np.random.choice(np.arange(n_X), n_X, replace=True)
            sample_y = np.random.choice(np.arange(n_Y), n_Y, replace=True)
            
            XvsAB_new = [self.XvsAB[i] for i in sample_x]
            YvsAB_new = [self.YvsAB[i] for i in sample_y]
            
            effect_size_b = cohend(XvsAB_new, YvsAB_new)
            effect_sizes_b.append(effect_size_b)
            
        effect_sizes_b = np.array(effect_sizes_b)
        effect_size, effect_size_ci_l, effect_size_ci_u = mean_confidence_interval(effect_sizes_b)
        
        return effect_size, effect_size_ci_l, effect_size_ci_u
            
        
        
class SWEAT(BaseWordVecComparison):
    def __init__(self, W, A, B, word_vectors_1, word_vectors_2):
        
        self.W = W
        self.A = A
        self.B = B
        self.word_vectors_1 = word_vectors_1
        self.word_vectors_2 = word_vectors_2
        
        self.word_vecotrs_name_1 = set(word_vectors_1.keys())
        self.word_vecotrs_name_2 = set(word_vectors_2.keys())
        
        # get the embeddings of each word in the sets
        self.W_arr_1, self.W_arr_2 = self._get_word_vectors(W)
        self.A_arr_1, self.A_arr_2 = self._get_word_vectors(A)
        self.B_arr_1, self.B_arr_2 = self._get_word_vectors(B)
        assert self._check_vector_set_pairs(self.A_arr_1, self.B_arr_1), 'Attribute sets (1) have not same number of words.'
        assert self._check_vector_set_pairs(self.A_arr_2, self.B_arr_2), 'Attribute sets (2) have not same number of words.'
        
        # compute association between target set against the two attribute sets
        self.WvsAB_1 = [self.s(w, self.A_arr_1, self.B_arr_1) for w in self.W_arr_1]
        self.WvsAB_2 = [self.s(w, self.A_arr_2, self.B_arr_2) for w in self.W_arr_2]
        
        
    def _get_word_vectors(self, words):
        '''
        Assign the word vector to each word in the list words
        
        Parameters:
        words : list of str, the list of words to transform into vectors
        '''
        words_vector_1 = np.vstack([self.word_vectors_1[w] for w in words if w in self.word_vecotrs_name_1])
        assert words_vector_1.shape[0]==len(words), "Some words have no embedding vector."
        
        words_vector_2 = np.vstack([self.word_vectors_2[w] for w in words if w in self.word_vecotrs_name_2])
        assert words_vector_2.shape[0]==len(words), "Some words have no embedding vector."
        
        return words_vector_1, words_vector_2

    
    def compute_score(self, approximate=True, n_iters=1000, return_test_statistic=False):
        
        WvsAB_12 = self.WvsAB_1 + self.WvsAB_2
        
        # get test statistic
        test_statistic = np.sum(self.WvsAB_1) - np.sum(self.WvsAB_2)

        # get p value
        if approximate:
            test_stats_rand = []
            for _ in range(n_iters):
                
                np.random.shuffle(WvsAB_12)
                WvsAB_1_new, WvsAB_2_new = WvsAB_12[:len(self.WvsAB_1)], WvsAB_12[len(self.WvsAB_1):]
                test_statistic_rand = np.sum(WvsAB_1_new) - np.sum(WvsAB_2_new)
           
                test_stats_rand.append(test_statistic_rand)
            test_stats_rand = np.array(test_stats_rand)
            
        else:
            raise NotImplementedError("Exact computation should be implemented.")


        # compute left/right tale p-value depending on the sign of test_statistic
        if test_statistic>0:
            p = (test_stats_rand > test_statistic).sum() / test_stats_rand.shape[0]
        else:
            p = (test_stats_rand < test_statistic).sum() / test_stats_rand.shape[0]
            
            
        if return_test_statistic:
            return test_statistic, p
        else:
            effect_size = cohend(self.WvsAB_1, self.WvsAB_2)
            return effect_size, p
        #return test_statistic, p


class CohesionTest(BaseWordVecComparison):
    def __init__(self, word_set_1, word_set_2, word_vectors):
        
        self.word_set_1 = word_set_1
        self.word_set_2 = word_set_2
        self.word_vectors = word_vectors
        self.word_vecotrs_name = set(word_vectors.keys())
        
        # get the embeddings of each word in the sets
        self.w_set_1_arr = self._get_word_vectors(word_set_1)
        self.w_set_2_arr = self._get_word_vectors(word_set_2)
        self._check_vector_set_pairs(self.w_set_1_arr, self.w_set_2_arr)
        
        # get size of word vectors
        self.n_w_set_1 = self.w_set_1_arr.shape[0]
        self.n_w_set_2 = self.w_set_2_arr.shape[0]
        
        
    def compute_cohesion(self, approximate=True, n_iters=1000):
        '''
        This computes the p-value for the cohesion of the pair of word sets and for each single word set.
        '''
        
        # get single group and average of group cohesion
        cohesion_1 = mean_cosine_similarity(self.w_set_1_arr)
        cohesion_2 = mean_cosine_similarity(self.w_set_2_arr)
        cohesion_both = np.mean([cohesion_1, cohesion_2])
        
        # compute p-values
        w_set_12_arr = np.vstack([self.w_set_1_arr, self.w_set_2_arr])
        
        cohesions_1_rand = []
        cohesions_2_rand = []
        cohesions_both_rand = []
        
        if approximate:
            for _ in range(n_iters):
                np.random.shuffle(w_set_12_arr)
                
                cohesion_1_rand = mean_cosine_similarity(w_set_12_arr[:self.n_w_set_1])
                cohesion_2_rand = mean_cosine_similarity(w_set_12_arr[self.n_w_set_1:])
                cohesion_both_rand = np.mean([cohesion_1_rand, cohesion_2_rand])
                
                cohesions_1_rand.append(cohesion_1_rand)
                cohesions_2_rand.append(cohesion_2_rand)
                cohesions_both_rand.append(cohesion_both_rand)
                
            cohesions_1_rand = np.array(cohesions_1_rand)
            cohesions_2_rand = np.array(cohesions_2_rand)
            cohesions_both_rand = np.array(cohesions_both_rand)
   
        else:
            NotImplementedError("Exact computation should be implemented.")
            
            
        # compute p-values
        p_1 = (cohesions_1_rand > cohesion_1).sum() / cohesions_1_rand.shape[0]
        p_2 = (cohesions_2_rand > cohesion_2).sum() / cohesions_2_rand.shape[0]
        p_both = (cohesions_both_rand > cohesion_both).sum() / cohesions_both_rand.shape[0]
        
        return p_1, p_2, p_both
        
        
        

        
       
    
        
        
        

