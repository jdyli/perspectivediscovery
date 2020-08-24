'''
Reformat data such that it suits the external source's topic model input
'''

import spacy
import pandas as pd
import os
'''
Format for JST, TAM:
docId [token1] [token2] ... [token n]
'''
def create_dataformat1(data, foldername, model):
    new_data = [' '.join([str(elem) for elem in i]) for i in data]
    filename = '../data/models/' + model + '/' + foldername + '/tokenized_data.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as filehandle:
        for item in range(0, len(new_data)):
            filehandle.write('%d %s\n' % (item, new_data[item]))

''' 
Format for VODUM:
#docs
docId token1:1 token2:0 ... token n:1/0
'''
def create_dataformat2(data, foldername, model):
    reformatted_data = [' '.join([str(elem) for elem in i]) for i in data]
    new_data = []
    opinion_words = ['VERB', 'ADJ', 'ADV']
    topic_words = ['NOUN', 'PROPN']
    nlp = spacy.load("en_core_web_sm")
    for doc in reformatted_data:
        new_string_list = []
        text = nlp(doc)
        for token in text:
            token2 = token
            if '_NEG' in token.text:
                token2 = nlp(token.text.split('_')[0])[0]
            if token2.pos_ in opinion_words:
                new_string = token.text + ':1 '
            elif token2.pos_ in topic_words:
                new_string = token.text + ':0 '
            elif token2.text == '.':
                new_string = '|'  # NON-EXISTENT
            else:
                new_string = ''
            new_string_list.append(new_string)
        new_data.append(''.join(new_string_list))

    filename = '../data/models/' + model + '/' + foldername + '/tokenized_data.dat'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as filehandle:
        filehandle.write('%d\n' % len(reformatted_data))
        for item in new_data:
            filehandle.write('%s\n' % item)
'''
Format for JTV
[token1] [token2] ... [token n]
'''
def create_dataformat3(data, foldername, model):
    print(foldername)
    new_data = [' '.join([str(elem) for elem in i]) for i in data]
    filename = '../data/models/' + model + '/' + foldername + '/tokenized_data_jtv.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as filehandle:
        for item in range(0, len(new_data)):
            filehandle.write('%s\n' % (new_data[item]))

'''
Format for LAM
[docid] [metadata] [full-text] [tokens]
'''
def create_dataformat4(perspective, articles, tokens, foldername, model):
    filename = '../data/models/' + model + '/' + foldername + '/tokenized_data_lam.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = pd.DataFrame()
    data['perspective'] = perspective
    data['article'] = articles
    data['tokens'] = tokens
    data.to_csv(filename, sep='\t', header=False, index=True)

'''
Read top 10 tokens per topic for JST
'''
def read_jst_toptopics(filename, twords_n=10):
    twords = []
    file_open = open(filename, "r")
    read_file = file_open.read().splitlines()
    for index, line in enumerate(read_file):
        if '_Topic' in line:
            subset_words = []
            for k in range(1, twords_n + 1):
                subset_words.append(read_file[index + k].split(' ')[0])
            twords.append(subset_words)
    file_open.close()
    return twords

'''
Read top 10 tokens per topic for TAM
'''
def read_tam_toptopics(filename, twords_n=5, topics_n=6):
    words_per_topic = 20
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.replace(' ', '') for line in lines] # remove empty spaces in file
    lines = [line.replace('\n', '') for line in lines]  # remove empty spaces in file
    totalwords = []
    topicwords = []
    backgroundwords = []
    get_topic_index = len(lines) # aspects will never reach the max line, so used as initialization
    get_background_index = len(lines)
    for index, line in enumerate(lines):
        if 'Background' in line:
            get_background_index = index
        elif 'Aspect' in line and index <= get_background_index + (words_per_topic * (topics_n + 2)): # or ('Topic' in line):
            subset_words = []
            for k in range(1, twords_n + 1):
                subset_words.append(lines[index + k].split(' ')[0])
            backgroundwords.append(subset_words)
        elif 'Topic' in line:
            get_topic_index = index
        elif 'Aspect' in line and index > get_topic_index: # or ('Topic' in line):
            subset_words = []
            for k in range(1, twords_n + 1):
                subset_words.append(lines[index + k].split(' ')[0])
            topicwords.append(subset_words)
        else:
            continue
        f.close()
    for i in range(0, len(backgroundwords)):
        totalwords.append(topicwords[i] + backgroundwords[i])
    return totalwords

'''
Read top 10 tokens per topic for VODUM
'''
def read_vodum_toptopics(foldername, twords_n=5):
    file_name_viewpoints = '../data/models/VODUM/' + foldername + '/model-01-final.vtwords'
    file_name_topics = '../data/models/VODUM/' + foldername + '/model-01-final.twords'

    # read the main perspectives
    twords_list = []
    twords_open = open(file_name_topics, "r")
    read_twords = twords_open.read().splitlines()
    for index, topic_line in enumerate(read_twords):
        if 'Topic' in topic_line:
            topic = []
            for k in range(1, twords_n + 1):
                topic.append(read_twords[index + k].split(' ')[0].lstrip('\t'))
            twords_list.append(topic)
    twords_open.close()

    vtwords_list = []
    file_open = open(file_name_viewpoints, "r")
    read_file = file_open.read().splitlines()
    for index, line in enumerate(read_file):
        if 'Viewpoint' in line:
            topic_number = int(line.split(',')[1].split(' ')[2].rstrip(':'))
            subset_words = []
            for k in range(1, twords_n + 1):
                subset_words.append(read_file[index + k].split(' ')[0].lstrip('\t'))
            vtwords_list.append(subset_words + twords_list[topic_number])
    file_open.close()
    return vtwords_list

'''
Read top 10 tokens per topic for LAM
'''
def read_lam_toptopics(filename):
    number_of_words = 5
    viewpoint_dict = {}
    viewpoint_list = []
    file_open = open(filename, "r")
    read_file = file_open.read().splitlines()
    for index, line in enumerate(read_file):
        if 'Topics' in line:
            topics = line.split(':')[1].lstrip(' ').split(' ')[:number_of_words]
            topic_number = line.split(':')[0].split(' ')[1]
            topic_index = index
        elif 'Viewpoint' in line and topic_index == index-1 or 'Viewpoint' in line and topic_index == index-2:
            viewpoints = line.split(':')[1].lstrip(' ').split(' ')[:number_of_words]
            viewpoint_number = line.split(':')[0].strip(' ').lstrip('\t').split(' ')[1]
            label = 't' + topic_number + 'vp' + viewpoint_number
            viewpoint_dict[label] = topics + viewpoints
            viewpoint_list.append(topics+viewpoints)
        else:
            continue
    file_open.close()
    return viewpoint_dict, viewpoint_list