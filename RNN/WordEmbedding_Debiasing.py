
# coding: utf-8

# # Operations on word vectors


import numpy as np
from w2v_utils import *


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v
    """
    
    distance = 0.0
    
    # Compute the dot product between u and v 
    dot = np.dot(u, v)
    # Compute the L2 norm of u
    norm_u = np.linalg.norm(u)
    
    # Compute the L2 norm of v 
    norm_v = np.linalg.norm(v)
    # Compute the cosine similarity
    cosine_similarity = dot / (norm_u * norm_v)
    ### END CODE HERE ###
    
    return cosine_similarity




def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    
    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    
    # Get the word embeddings v_a, v_b and v_c
    e_a, e_b, e_c = word_to_vec_map[word_a],word_to_vec_map[word_b],word_to_vec_map[word_c]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None
    cosine_sim_orig = e_b - e_a
    # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:        
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c] :
            continue
        
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c) 
        cosine_sim = cosine_similarity(cosine_sim_orig , (word_to_vec_map[w] - e_c))
        
        # If the cosine_sim is more than the max_cosine_sim seen so far,
            # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word


def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.
    
    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    
    # Select word vector representation of "word". Use word_to_vec_map.
    e = word_to_vec_map[word]
    
    # Compute e_biascomponent using the formula give above.
    e_biascomponent = np.dot(e,g)/(np.linalg.norm(g)**2) * g
 
    # Neutralize e by substracting e_biascomponent from it 
    # e_debiased should be equal to its orthogonal projection.
    e_debiased = e - e_biascomponent
    
    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    
    # Step 1: Select word vector representation of "word". Use word_to_vec_map.
    w1, w2 = pair[0],pair[1]
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    # Step 2: Compute the mean of e_w1 and e_w2
    mu = (e_w1 + e_w2)/2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis
    mu_B = (np.dot(mu, bias_axis)/(np.linalg.norm(bias_axis)**2)) * bias_axis
    mu_orth = mu - mu_B

    # Step 4: Compute e_w1B and e_w2B
    e_w1B = (np.dot(e_w1, bias_axis)/np.linalg.norm(bias_axis)**2) * bias_axis
    e_w2B = (np.dot(e_w2, bias_axis)/np.linalg.norm(bias_axis)**2) * bias_axis
        
    # Step 5: Adjust the Bias part of e_w1B and e_w2B 
#     np.sqrt(abs(1-np.linalg.norm(mu_orth)**2)) * ((e_w1B -mu_B)/np.linalg.norm(e_w1B-mu_B))


    corrected_e_w1B = np.sqrt(abs(1 - np.linalg.norm(mu_orth)**2)) * ((e_w1B - mu_B)/np.linalg.norm(e_w1B - mu_B))
    corrected_e_w2B = np.sqrt(abs(1 - np.linalg.norm(mu_orth)**2)) * ((e_w2B - mu_B)/np.linalg.norm(e_w2B - mu_B))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
                                                                
    
    return e1, e2


