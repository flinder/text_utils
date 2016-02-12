from __future__ import unicode_literals, print_function, division

from nltk.corpus import stopwords
#from spacy.en import English
#import Stemmer
from pprint import pprint
import re
import numpy as np
import io
import scipy.sparse as ssp
import logging
from gensim import corpora
from gensim import matutils
from gensim import models
import sys

class Cleaner(object):

    def __init__(self):
        '''Load several regex used for text cleaning and redaction
        '''
        self.re_apo = re.compile(r'\s(t|d|s|ll|m|ve|nt|re)(\s|$|!|\?|\.|:|,)')
        self.re_white = re.compile(r'(\s)+|^\s|\s$')
        self.re_url = re.compile(r'http:[^\s$,:]+')
        self.re_hashtag = re.compile(r'(^|\s)#[^\s$\n]+')
        self.re_handle = re.compile(r'@[^\s$\n]+')
        self.re_exclude = re.compile(r'[^A-Za-z @#_]')
        #self.re_exclude = re.compile(r'[\.,\(\)-+=!\$%\^&\*<>"\';:]')        
        self.re_linebreak = re.compile(r'\n')
        self.re_hashsep = re.compile(r'(#)( )([^\s]+)')
        self.re_at = re.compile(r'@(\s|$)')

    def pre_clean(self, string):
        string = self.re_linebreak.sub(r'', string)
        string = self.re_apo.sub(r"'\1\2", string)
        string = self.re_at.sub(r"at\1", string)
        string = self.re_white.sub(r' ', string)
        if string.isspace() or string == '':
            string = '__empty__'
     
        return string

    def post_clean(self, string): 
        string = self.re_exclude.sub('', string) 
        string = self.re_hashsep.sub(r'\1\3', string)
        string = self.re_white.sub(r' ', string)
        if string.isspace() or string == '':
            string = '__empty__'

        return string

def n_grams(text, parser, n = 1, stemmer = None, stopwords = None,
        lemmatize=True):
    '''Generate list of n-grams from text 
    
    Arguments:
    sentence (string): Sentence to be converted
    parser (spacy.English): Spacy NLP model
    n (int): the n in n-gram
    stemmer: a stemmer to be used, if None no stemming is done
    stopwords: A list of stopwords to be removed, if None no stopwords are
       removed
    '''

    if lemmatize and stemmer is not None:
        raise ValueError("Choose either lemmatization or stemming")

    doc = parser(text) 
    n_gram_list = []
    tokens = []
    if stopwords is not None:
        stopwords = set(stopwords)

    if lemmatize:
        for sentence in doc.sents:
            
            for token in sentence:
                if stopwords is not None:
                    if token.lemma_.lower() in stopwords:
                        continue
                tokens.append(token.lemma_.lower())
    else:
        for token in doc:
            if stopwords is not None:
                if token.orth_.lower() in stopwords:
                    continue
            tokens.append(stemmer.stemWord(token.orth_.lower()))

    # Generate n-grams
    if n > 1:
    	for index, token in enumerate(tokens):
            try:
            	gram_tokens = [tokens[i] for i in range(index, (index + n))]
            except IndexError:
            	break 
            gram = '_'.join(gram_tokens)
            n_gram_list.append(gram)
    else:
        n_gram_list = tokens

    return n_gram_list

def extract_tweet(line, field):
    '''Extract tweet from tab separated line

    '''
    fields = line.split('\t')
    try:
        return fields[field]
    except IndexError:
        return ''

def save_sparse_csc(filename, array):
    ''' Save a sparse csc matrix to file
    '''
    np.savez(filename, data=array.data, indices=array.indices,
            indptr=array.indptr, shape=array.shape)

def load_sparse_csc(filename):
    ''' Load a sparse matrix that was saved with save_sparse_csc()
    '''
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shpe = loader['shape']
    return ssp.csc_matrix((data, indices, indptr), shape=shpe)

def store_matrix(vocabulary, sparse_matrix, filename):
    ''' Store a sparse matrix and vocabulary to file

    '''
    save_sparse_csc(filename, sparse_matrix)
    voc_file = filename + '_vocabulary.txt'
    with io.open(voc_file, 'w+', encoding = 'utf-8') as outfile:
        outfile.write('\n'.join(vocabulary))


class TweetCorpus(object):
    '''
    A stream object for tweets stored in a plain text file
    
    Arguments:
    ----------
    dictionary: Gensim dictionary with terms to include
    text_input (str): Input file containing the tweets
    use_tfidf (bool): Should tf-idf scores be used instead of wordcounts
    status (bool): Should status updates be printed
    '''
    def __init__(self, dictionary, text_input, limit, use_tfidf):
        self.dictionary = dictionary
        self.text_input  = io.open(text_input, 'r', encoding='utf-8')
        self.limit = limit
        self.use_tfidf = use_tfidf

    def __iter__(self):
        if self.use_tfidf: 
            self.tfidf = models.TfidfModel(dictionary=dictionary)     
        for index, line in enumerate(self.text_input): 
            if index % 1000 == 0:
                print("\rProcessed {} tweets".format(index), end='')
                sys.stdout.flush()
            if self.limit is not None and index > self.limit:
                break
            if self.use_tfidf:
                yield self.tfidf[self.dictionary.doc2bow(line.split())]
            else:
                yield self.dictionary.doc2bow(line.split())
        print('\n' + unicode(index))


class CorpusStream(object):
    '''
    A stream object for text stored in a text file (one line per doc)

    Arguments:
    ----------
    dictionary: Gensim dictionary with terms to include
    text_input (str): Input file
    use_tfidf (bool): Should tf-idf scores be used instead of wordcounts
    status (bool): Should status updates be printed
    '''
    def __init__(self, dictionary, text_input, status, use_tfidf):
        self.dictionary = dictionary
        self.text_input  = io.open(text_input, 'r', encoding='utf-8')
        self.use_tfidf = use_tfidf
        self.status = status

    def __iter__(self):
        if self.use_tfidf: 
            self.tfidf = models.TfidfModel(dictionary=self.dictionary)     
        for index, line in enumerate(self.text_input): 
            if index % 1000 == 0 and self.status:
                print("\rProcessed {} documents".format(index), end='')
                sys.stdout.flush()
            if self.use_tfidf:
                yield self.tfidf[self.dictionary.doc2bow(line.split())]
            else:
                yield self.dictionary.doc2bow(line.split())

        print('\n' + unicode(index))




def tdm_from_stream(dict_file, text_file, use_tfidf, limit=None):
    '''
    Generate a term document matrix from a dictionary and a text input
    
    Arguments:
    ----------  
    dict_file (str): pkl file containing the dictionary
    text_file (str): text file containing one tweet per line
    use_tfidf (bool): Sould tfidf scores be used instead of wordcounts
    limit (int): How many tweets should be processed. If None all are used

    Returns:
    ----------
    A scipy sparese column matrix (csc) of dimensions (num_terms x num_tweets)
    '''
    dictionary = corpora.Dictionary.load(dict_file)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n = None)

    stream = TweetCorpus(dictionary=dictionary, text_input=text_file, 
                         limit=limit, use_tfidf=use_tfidf)
    features = matutils.corpus2csc(stream, num_terms=len(dictionary), dtype=int,
                                   printprogress=0, num_docs=dictionary.num_docs)
    return features




