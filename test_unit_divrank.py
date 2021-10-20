"""
Created on Thu Aug  6 10:26:16 2021
@author: Dr. Deepak Kumar
"""
import cumulative_divrank
import cum_cdvrank_main
import pytest

cls_precosess = cum_cdvrank_main.cls_preprocess_data
@pytest.fixture
def text_normal_input():
    '''Provide  the normal test case Input'''
    filename = "dataset/divrank_data.txt"
    return filename

@pytest.fixture
def text_random_input():
    '''Provide  the test case Input'''
    file_list = ["dataset/divrank_data_test.txt", "dataset/divrank_data_test1.text",
                 "dataset/divrank_data_test1.doc"]
    return file_list

def test_txt_to_keywords_normal_input(text_normal_input):
    '''Unit test case for DivRank with appropriate Input'''
    key_list = cum_cdvrank_main.txt_to_keywords(text_normal_input, window_size=10,
                                                nb_keywords=50, lambda_=0.5, alpha=0.5)
    assert not key_list.empty
def test_txt_to_keywords_wrong_input(text_random_input):
    '''Unit test case for DivRank with wrong or blank Input'''
    for filename in text_random_input:
        key_list = cum_cdvrank_main.txt_to_keywords(filename, window_size=10,
                                                    nb_keywords=50, lambda_=0.5, alpha=0.5)
    assert key_list.empty
def test_txt_to_sentences_normal_input(text_normal_input):
    '''Unit test case for others function appropriate Input'''
    txt_sentences = cls_precosess.txt_to_sentences(text_normal_input)
    assert txt_sentences
    clean_text = cls_precosess.clean_text_simple_by_sents(txt_sentences, remove_stopwords=True,
                                                          pos_filtering=True)
    assert clean_text
    tuples_words = (list(cls_precosess.terms_to_graph_sents(clean_text, window_size=10,
                                                            stopping_end_of_line=False).keys()))
    assert tuples_words

    div_graph = cum_cdvrank_main.unweighted_graph(tuples_words)
    assert div_graph
    div_rank = cumulative_divrank.com_divrank(div_graph, lambda_=0.5, alpha=0.5)
    assert div_rank
def test_case_file_with_nocontent(text_random_input):
    '''Unit test case for with others function wrong or blank Input'''
    for filename in text_random_input:
        txt_sentences = cls_precosess.txt_to_sentences(filename)
        assert not txt_sentences
        clean_text = cls_precosess.clean_text_simple_by_sents(txt_sentences, remove_stopwords=True,
                                                              pos_filtering=True)
        assert not clean_text
        tuples_words = (list(cls_precosess.terms_to_graph_sents(clean_text, window_size=10,
                                                                stopping_end_of_line=False).keys()))
        assert not tuples_words
        cum_rank = cum_cdvrank_main.unweighted_graph(tuples_words)
        assert not cum_rank
        div_rank = cumulative_divrank.com_divrank(cum_rank, lambda_=0.5, alpha=0.5)
        assert not div_rank