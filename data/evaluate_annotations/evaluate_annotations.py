import krippendorff as krippendorff
import pandas as pd
import krippendorff
import numpy as np

pd.options.mode.chained_assignment = None

def get_perspectives_based_on_sample(sample_data, main_data):
    sample_data = sample_data.set_index('web-scraper-order')
    main_data = main_data.set_index('web-scraper-order')
    sub_data = main_data[main_data.index.isin(sample_data.index)]
    sub_data['annotator'] = sample_data['perspective']
    sub_data['id'] = sample_data['title']
    return sub_data

def get_stance_based_on_sample(sample_data, main_data):
    sample_data = sample_data.set_index('web-scraper-order')
    main_data = main_data.set_index('web-scraper-order')
    sub_data = main_data[main_data.index.isin(sample_data.index)]
    sub_data['annotator'] = sample_data['stance']
    return sub_data

def stance_to_numeric(x):
    if x=='for':
        return int(1)
    if x=='against':
        return int(0)

def score_to_numeric(x):
    if x=='[1]':
        return int(1)
    if x=='[2]':
        return int(2)
    if x=='[3]':
        return int(3)
    if x=='[11]':
        return int(11)
    if x=='[14]':
        return int(14)
    if x=='[15]':
        return int(15)
    if x=='[16]':
        return int(16)
    if x=='[17]':
        return int(17)
    if x=='[19]':
        return int(19)
    if x=='[25]':
        return int(25)
    if x=='[7]':
        return int(7)

def batch_to_full(filename_list):
    annotations = pd.read_csv(filename_list[0],
                names=['Unnamed: 0', 'web-scraper-order', 'web-scraper-start-url', 'article',
                       'title', 'stance', 'perspective'])

    for i in range(1, len(filename_list)):
        annotations_next = pd.read_csv(filename_list[i],
                                                names=['Unnamed: 0', 'web-scraper-order', 'web-scraper-start-url', 'article',
                                                         'title', 'stance', 'perspective'])
        annotations = pd.concat([annotations, annotations_next])
    annotations_total = annotations.reset_index(drop=True)
    annotations_total['perspective'] = annotations_total['perspective'].apply(score_to_numeric)
    annotations_total['stance'] = annotations_total['stance'].apply(stance_to_numeric)
    return annotations_total

def calculate_agreement(main_annotator, second_annotator, attribute_type):
    if attribute_type == 'perspective':
        main_annotator_sub = get_perspectives_based_on_sample(second_annotator, main_annotator)
    else:
        main_annotator_sub = get_stance_based_on_sample(second_annotator, main_annotator)
    reliability_data = [main_annotator_sub[attribute_type], main_annotator_sub['annotator']]
    krippendorff_alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
    print(attribute_type, ': ', krippendorff_alpha)

main_annotator_abortion1 = pd.read_csv('abortion_debateorg816_mainannotator.csv')
main_annotator_abortion1['perspective'] = main_annotator_abortion1['perspective'].apply(score_to_numeric).astype(float)
main_annotator_abortion1['stance'] = main_annotator_abortion1['stance'].apply(stance_to_numeric).astype(float)

main_annotator_abortion2 = pd.read_csv('abortion_debateorg600_mainannotator.csv')
main_annotator_abortion2['perspective'] = main_annotator_abortion2['perspective'].apply(score_to_numeric).astype(float)
main_annotator_abortion2['stance'] = main_annotator_abortion2['stance'].apply(stance_to_numeric).astype(float)

annotator1_list = ['annotated/abortion_with17/abortion_debateorg816_annotated_sample2_0_11.csv',
            'annotated/abortion_with17/abortion_debateorg816_annotated_sample2_30_60.csv',
            'annotated/abortion_with17/abortion_debateorg816_annotated_sample2_60_90.csv',
            'annotated/abortion_with17/abortion_debateorg816_annotated_sample2_90_120.csv',
            'annotated/abortion_with17/abortion_debateorg816_annotated_sample2_120_150.csv',
            'annotated/abortion_with17/abortion_debateorg816_annotated_sample2_150_180.csv',
            'annotated/abortion_with17/abortion_debateorg816_annotated_sample2_180_210.csv',
            'annotated/abortion_with17/abortion_debateorg816_annotated_sample2_11_30.csv']
annotator1_data = batch_to_full(annotator1_list)

annotator2_list = ['annotated/abortion_without17/abortion_debateorg600_annotated_sample1_0_30.csv',
            'annotated/abortion_without17/abortion_debateorg600_annotated_sample1_30_60.csv',
            'annotated/abortion_without17/abortion_debateorg600_annotated_sample1_60_90.csv',
            'annotated/abortion_without17/abortion_debateorg600_annotated_sample1_90_120.csv',
            'annotated/abortion_without17/abortion_debateorg600_annotated_sample1_120_150.csv',
            'annotated/abortion_without17/abortion_debateorg600_annotated_sample1_150_180.csv',
            'annotated/abortion_without17/abortion_debateorg600_annotated_sample1_180_200.csv']
annotator2_data = batch_to_full(annotator2_list)

print('----- Interrater-reliability score on perspective and stance between main annotator and external annotator -----')
print('Abortion ---- Annotator 1')
calculate_agreement(main_annotator_abortion1, annotator1_data, 'perspective')
calculate_agreement(main_annotator_abortion1, annotator1_data, 'stance')

print('Abortion ---- Annotator 2')
calculate_agreement(main_annotator_abortion2, annotator2_data, 'perspective')
calculate_agreement(main_annotator_abortion2, annotator2_data, 'stance')
