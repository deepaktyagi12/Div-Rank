"""
Created on Thu Aug  5 20:26:16 2021
@author: Dr. Deepak Kumar
"""
import nltk
from nltk.corpus import stopwords
import textacy
import argparse
import cumulative_divrank
import logging
LOGGER = logging.getLogger(__name__)
import os
import networkx as nx
import itertools
import pandas as pd
import configparser

# import textacy.keyterms
class  cls_preprocess_data:
    '''Class to read and preprocess the text data'''
    def txt_to_sentences(file_title):
        '''Convert text to sentences, keyword with Part of speech'''
        if  os.path.isfile(file_title):
            with open(file_title, "r", encoding='utf-8') as text:
                text = str(text.read())
                if text:
                    text = text.lower()
                    text = textacy.preprocess_text(text, fix_unicode=False, lowercase=False,
                                                   transliterate=False, no_urls=False,
                                                   no_phone_numbers=False,
                                                   no_currency_symbols=False, no_emails=True,
                                                   no_numbers=False, no_punct=True,
                                                   no_contractions=False, no_accents=False)
                    nlp_text = textacy.Doc(text).pos_tagged_text
                    return nlp_text
                else:
                    LOGGER.error("Please check the file name!")
                    return {}
        else:
            # raise InvalidDocumentType('File Not FoundError')
            LOGGER.error('File Not Found Error')
            return {}
    def clean_text_simple_by_sents(tok_sent_tagged, remove_stopwords=True, pos_filtering=True):
        '''Return the clean text by removing the stopwords and filtering'''
        cleaned_text = []
        stop_words = stopwords.words('english')
        for i, phrase in enumerate(tok_sent_tagged):
            tokens = []
            for words in enumerate(phrase):
                if words:
                    if pos_filtering:
                        if (words[1][1] == 'NOUN' or words[1][1] == 'ADJ'):
                            tokens.append(words[1][0])
                    else:
                        tokens.append(words[1][0])
            if remove_stopwords:
                tokens = [token for token in tokens if token not in stop_words]
                # apply Porter's stemmer
            stemmer = nltk.stem.PorterStemmer()
            tokens_stemmed = list()
            for token in tokens:
                tokens_stemmed.append(stemmer.stem(token))
            tokens = tokens_stemmed
            cleaned_text.append(tokens)
        return cleaned_text
    def terms_to_graph_sents(clean_txt_sents, window_size, stopping_end_of_line=0):
        '''Create the GRAPH for clean input '''
        from_to = {}
        if not stopping_end_of_line:
            extended_clean_txt_sents = []
            for sublist in clean_txt_sents:
                extended_clean_txt_sents.extend(sublist)
            clean_txt_sents = [extended_clean_txt_sents]
        for key, sents in enumerate(clean_txt_sents):
            len_sents = len(sents)
            # create initial complete graph (first w terms)
            terms_temp = sents[0:min(window_size, len_sents)]
            indexes = list(itertools.combinations(range(min(window_size, len_sents)), r=2))
            new_edges = []
            for my_tuple in indexes:
                new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
            for new_edge in new_edges:
                if new_edge:
                    if new_edge in from_to:
                        from_to[new_edge] += 1
                    else:
                        from_to[new_edge] = 1
            if window_size <= len_sents:
                # then iterate over the remaining terms
                for i in range(window_size, len_sents):
                    considered_term = sents[i] # term to consider
                    terms_temp = sents[(i-window_size+1):(i+1)]
                    # all terms within sliding window
                    # edges to try
                    candidate_edges = []
                    for prob in range(window_size-1):
                        candidate_edges.append((terms_temp[prob], considered_term))
                    for try_edge in candidate_edges:
                        if try_edge[1] != try_edge[0]:
                        # if not self-edge
                            # if edge has already been seen, update its weight
                            if try_edge in from_to:
                                from_to[try_edge] += 1
                            # if edge has never been seen, create it and assign it a unit weight
                            else:
                                from_to[try_edge] = 1
        return from_to

def unweighted_graph(tuples_words_unweighted):
    ''' Generate UNWEIGHTED GRAPH'''
    un_graph = nx.Graph()
    un_graph.add_edges_from(tuples_words_unweighted)
    return un_graph

def order_dict_best_keywords(g_core_number, no_keys_terms_needed):
    '''Generate and order a list of best keywords,
    the number is specified by "no_keys_terms_needed"'''
    k_core_keyterms = sorted(g_core_number, key=g_core_number.get, reverse=True)
    Kcore_values = [g_core_number[x] for x in k_core_keyterms[:no_keys_terms_needed]]
    return k_core_keyterms, Kcore_values

def txt_to_keywords(filename, window_size, nb_keywords, lambda_, alpha):
    ''' Extract and Rank keywords from text document or corpus documents using a graph approach.'''
    if filename:
        cls_precosess = cls_preprocess_data
        txt_sentences = cls_precosess.txt_to_sentences(filename)
        clean_txt_sents = cls_precosess.clean_text_simple_by_sents(txt_sentences,
                                                                   remove_stopwords=True,
                                                                   pos_filtering=True)
        tuples_words_sents = list(cls_precosess.terms_to_graph_sents(clean_txt_sents, window_size,
                                                                     stopping_end_of_line=False).keys())
        G = unweighted_graph(tuples_words_sents)
        # DivRank=textacy.keyterms.rank_nodes_by_divrank(G,lambda_=0.5, alpha=0.25)
        DivRank = cumulative_divrank.com_divrank(G, lambda_, alpha)
        DivRank_keyTerms, DivRank_values = order_dict_best_keywords(DivRank, nb_keywords)
        df_results = pd.DataFrame(columns=['Div_Rank_KeyTerms', 'DR_values'])
        df_results['Div_Rank_KeyTerms'] = DivRank_keyTerms[:nb_keywords]
        df_results['DR_values'] = DivRank_values[:nb_keywords]
        print(df_results[0:50])
        if not df_results.empty:
            out_file_path = filename.split('.')[0]
            out_file_path = out_file_path+"_output.csv"
            df_results.to_csv(out_file_path)
        return df_results
if __name__ == '__main__':
    config_file = "config.cfg"
    config = configparser.ConfigParser()
    config.read(config_file)
    file_path1 = config["data_path"].get("filepath")
    window_size = config["divrank_param"].getint("window_size")
    nb_keywords = config["divrank_param"].getint("nb_keywords")
    # per_vect = config["divrank_param"].get("r")
    lambda_ = config["divrank_param"].getfloat("lambda")
    alpha = config["divrank_param"].getfloat("alpha")
    max_iter = config["divrank_param"].getint("max_iter")
    diff = config["divrank_param"].getfloat("diff")
    tol = config["divrank_param"].getfloat("tol")
    txt_to_keywords(file_path1, window_size, nb_keywords, lambda_, alpha)
    