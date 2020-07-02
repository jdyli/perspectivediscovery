import gensim
import gensim.corpora as corpora
import spacy
import pandas as pd
import contractions
import nltk
from spacy.symbols import nsubj, dobj, prep, amod, conj, ADV, VERB
import statistics
import re
from tqdm import tqdm
from nltk.corpus import sentiwordnet
from nltk.corpus import wordnet
from spellchecker import SpellChecker

stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(['abortion816', 'u', 'ur', 'huh', 'le', 'ha', 'wa', 'doe', 'etc', 'abortion', 'abort'])
negation_stopwords = {'not', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'neither',
                      'never', 'nevertheless'}
final_stop_words = set([word for word in stop_words if word not in negation_stopwords])

'''Set up of the model'''


def prepare_model(documents):
    '''Start corpus'''
    dictionary = corpora.Dictionary(documents)  # Create Dictionary
    texts = documents  # Create Corpus
    corpus = [dictionary.doc2bow(text) for text in texts]  # Term Document Frequency
    return (corpus, dictionary)


'''Start baseline'''


def tokenize(data):
    spell = SpellChecker(distance=1)
    spell.word_frequency.load_words(['pro-life', 'pro-choice'])
    wnl = nltk.WordNetLemmatizer()

    data_new = []
    for article in tqdm(data):
        article_new = []
        sentences = re.split(r'[.:;!?]', article)
        for sentence in sentences:
            i = 0
            split_sentence = list(sent_to_words(sentence))  # lowercase, punctuation, tokenize
            while i < len(split_sentence):
                word = split_sentence[i]
                misspelled = spell.unknown([word])
                if word in misspelled:
                    word = spell.correction(word)
                lemmatized_word = [wnl.lemmatize(word)]
                if lemmatized_word[0] not in final_stop_words:  # stopword removal
                    i = i + 1
                    article_new.extend(lemmatized_word)
                else:
                    i = i + 1
        data_new.append(article_new)
    return data_new


def text_correction(data):
    text_corrected = pd.DataFrame(columns=['article_new'])
    for i in range(0, len(data)):
        text_corrected = text_corrected.append({'article_new': contractions.fix(data.iloc[i])}, ignore_index=True)
    return text_corrected['article_new']


'''End baseline'''


def remove_extremes(tokenized_data):
    extremes = []
    # frequency distributed
    tokenized_list = list([a for b in tokenized_data for a in b])  # TODO also for other models
    fdist = nltk.FreqDist(tokenized_list)
    fdist_mean = statistics.mean(fdist.values())
    fdist_std = statistics.stdev(fdist.values(), xbar=fdist_mean)
    for item in fdist.items():
        if (item[1] > fdist_mean + (6 * fdist_std)):  # or (item[1] < fdist_mean - (2*fdist_std)):
            extremes.append(item[0])
    extremes.extend(fdist.hapaxes())

    tokenized_new = []
    for tokens in tokenized_data:
        tokenized = [x for x in tokens if x not in extremes]
        tokenized_new.append(tokenized)
    return tokenized_new


def zipslaw(data, frequentp):
    documents_n = len(data)
    tokenized_list = list([a for b in data for a in b])
    fdist = nltk.FreqDist(tokenized_list)
    frequent_tokens_n = int(documents_n * frequentp)

    frequent_tokens = fdist.most_common(frequent_tokens_n)

    tokens_to_remove_list = []
    for token in frequent_tokens:
        if token[1] >= frequent_tokens_n:
            tokens_to_remove_list.append(token[0])
    tokens_to_remove_list.extend(fdist.hapaxes())

    tokenized_new = []
    for tokens in data:
        tokenized = [x for x in tokens if x not in tokens_to_remove_list]
        tokenized_new.append(tokenized)
    return tokenized_new


def sent_to_words(sentences):
    # for sentence in sentences:
    return gensim.utils.simple_preprocess(str(sentences), deacc=True)  # deacc=True removes punctuations


def bigrams_and_unigrams(data):
    new_data = []
    for article in data:
        bigrams = [article[i] + '_' + article[i + 1] for i in range(0, len(article) - 1)]
        bigrams.extend(article)
        new_data.append(bigrams)
    return new_data


def trigrams_bigrams_unigrams(data):
    new_data = []
    for article in data:
        bigrams = [article[i] + '_' + article[i + 1] for i in range(0, len(article) - 1)]
        bigrams.extend(article)
        tigrams = [article[i] + '_' + article[i + 1] + '_' + article[i + 2] for i in range(0, len(article) - 2)]
        tigrams.extend(bigrams)
        new_data.append(tigrams)
    return new_data


def convert_to_simple_postag(word):
    word_tag = nltk.pos_tag([word])[0][1]
    if word_tag.startswith('J'):
        return wordnet.ADJ
    elif word_tag.startswith('N'):
        return wordnet.NOUN
    elif word_tag.startswith('R'):
        return wordnet.ADV
    elif word_tag.startswith('V'):
        return wordnet.VERB
    return None


def remove_nonsentiments(data):
    new_data = []
    for article in data:
        new_article = []
        for token in article:
            if '_' not in token or 'not' not in token or 'no' not in token:
                pos = convert_to_simple_postag(token)
                if pos != wordnet.NOUN:
                    synsets = wordnet.synsets(token, pos=pos)
                    if synsets:
                        sentiword = sentiwordnet.senti_synset(synsets[0].name())
                        if sentiword.pos_score() != 0 or sentiword.neg_score() != 0:
                            new_article.append(token)
                    else:
                        continue
                else:
                    new_article.append(token)
            else:
                new_article.append(token)
        new_data.append(new_article)
    return new_data


def negation(data):
    new_data = []
    for article in data:
        new_article = []
        counter = 0
        while counter < len(article) - 1:
            if 'not' in article[counter] or 'cannot' in article[counter] or 'no' in article[counter]:
                new_article.append(article[counter] + '_' + article[counter + 1])
                counter += 2
            else:
                new_article.append(article[counter])
                counter += 1
        if counter == len(article) - 1:
            new_article.append(article[counter])
        new_data.append(new_article)
    return new_data


def antonyms(data):
    new_data = []
    for article in data:
        new_article = []
        i = 0
        while i < len(article) - 2:
            new_token = article[i]
            if article[i] == 'not' or article[i] == 'no' or article[i] == 'never' or article[i] == 'none':
                synset_set = wordnet.synsets(article[i + 1])
                for syn in synset_set:
                    for lm in syn.lemmas():
                        if lm.antonyms() and lm.name() == article[i + 1]:
                            new_token = lm.antonyms()[0].name()
                            i = i + 1
            new_article.append(new_token)
            i = i + 1
        new_data.append(new_article)
    return new_data

'''Other functions tried, but did not improve the model'''
# def build_bigrams(texts):
#     bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
#     bigram_mod = gensim.models.phrases.Phraser(bigram)
#     return bigram_mod
#
# def make_bigrams(texts, bigram_mod):
#     return [bigram_mod[doc] for doc in texts]

# def enforce_stance(data, perspectives, maxpro_n):
#     for i in range(0, len(data)):
#         if perspectives[i][0] <= maxpro_n:
#             data[i].extend(['legal'])
#         else:
#             data[i].extend(['illegal'])
#     return data

# def only_nouns(data):
#     new_data = []
#     for article in data:
#         new_article = []
#         for token in article:
#             pos = convert_to_simple_postag(token)
#             if pos == wordnet.NOUN:
#                 new_article.append(token)
#             else:
#                 continue
#         new_data.append(new_article)
#     return new_data

# def get_stance_target_tokens(text_list):
#     nlp = spacy.load("en_core_web_sm")
#     complete_data = []
#     wnl = nltk.WordNetLemmatizer()
#
#     for text in text_list:
#         # text = text.lower()
#         doc = nlp(" ".join(text))
#         text_data = []
#         for token in doc:
#             if (token.dep == nsubj) and (token.head.pos == VERB): # Direct subject rule
#                 target = str(token.text)
#                 opinion = str(token.head)
#             elif token.dep == dobj: # Direct object rule
#                 target = str(token.text)
#                 opinion = str(token.head)
#             elif token.dep == amod: # Adjectival modifier rule
#                 target = str(token.head)
#                 opinion = str(token.text)
#
#                 for i in token.head.children:
#                     if (i.dep == prep): # Prepositional object rule
#                         for child in i.children:
#                             child_target = wnl.lemmatize(str(child))
#                             child_opinion = str(token.text)
#
#                             child_pair = child_target + " " + child_opinion
#                             text_data.append(child_pair)
#                 for child in token.children: # Recursive modifiers rule
#                     if (child.dep == conj):
#                         child_target = wnl.lemmatize(str(token.head))
#                         child_opinion = str(child.text)
#
#                         child_pair = child_target + " " + child_opinion
#                         text_data.append(child_pair)
#             else:
#                 continue
#             pair = wnl.lemmatize(target) + " " + opinion
#             text_data.append(pair)
#         text_data_string = " ".join(text_data)
#         complete_data.append(text_data_string.split(" "))
#     return complete_data

# def semantic_similarity(data):
#     nlp = spacy.load("en_core_web_md")  # make sure to use larger models
#
#     unique_tokens = " ".join(set([a for b in data for a in b]))
#     unique_tokens_nlp = nlp(unique_tokens)
#
#     for article in tqdm(data):
#         article_nlp = nlp(" ".join(article))
#         for token in article_nlp:
#             for unique in unique_tokens_nlp:
#                 if token.text != unique.text and token.has_vector and unique.has_vector:
#                     if token.similarity(unique) > 0.8:
#                         article.extend([unique.text])
#     return data

# def bigrams(data):
#     unlist_comments = [item for items in data for item in items]
#     bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(unlist_comments)
#     bigram_freq = bigramFinder.ngram_fd.items()
#     bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram', 'freq']).sort_values(by='freq',
#                                                                                               ascending=False)
#     bigramFreqTable = bigramFreqTable.reset_index(drop=True)
#     return bigramFreqTable
#
# # function to filter for ADJ/NN bigrams
# def rightTypes(ngram):
#     if '-pron-' in ngram or '' in ngram or ' ' in ngram or 't' in ngram:
#         return False
#     for word in ngram:
#         if word in final_stop_words:
#             return False
#     acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
#     second_type = ('NN', 'NNS', 'NNP', 'NNPS')
#     tags = nltk.pos_tag(ngram)
#     if tags[0][1] in acceptable_types and tags[1][1] in second_type:
#         return True
#     else:
#         return False
#
# def tokens_to_bigrams(tokenized_data):
#     bigram_data = bigrams(tokenized_data)
#     filtered_bi = bigram_data[bigram_data.bigram.map(lambda x: rightTypes(x))]
#     filtered_bi_list = list(filtered_bi['bigram'])
#
#     new_articles = []
#     for article in tokenized_data:
#         article_tokens = []
#         counter = 0
#         while counter < len(article)-1:
#             bigram_tuple = (article[counter], article[counter+1])
#             if bigram_tuple in filtered_bi_list:
#                 article_tokens.append(article[counter] + '_' + article[counter+1])
#                 counter += 2
#             else:
#                 article_tokens.append(article[counter])
#                 counter += 1
#         new_articles.append(article_tokens)
#     return new_articles

# def trigrams_bigrams(data):
#     new_data = []
#     for article in data:
#         bigrams = [article[i] + '_' + article[i + 1] for i in range(0, len(article) - 1)]
#         trigrams = [article[i] + '_' + article[i + 1] + '_' + article[i+2] for i in range(0, len(article) - 2)]
#         trigrams.extend(bigrams)
#         new_data.append(trigrams)
#     return new_data
