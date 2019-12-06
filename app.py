from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask import jsonify
import pandas as pd
import numpy as np
import os.path
import csv

app = Flask(__name__)
CORS(app, resources=r'*')
dictionary = pd.read_excel('lexicon.xlsx', sheet_name='V')
dictionary = dictionary.replace(np.nan, '', regex=True)
print(dictionary.columns)
sentencesToReturn = []
mappings = dict()
mappings['Common Noun'] = 'CN'
mappings['Proper Noun'] = 'PN'
mappings['Verb'] = 'V'
mappings['Adverb'] = 'AV'
mappings['Adjective'] = 'AJ'
mappings['Preposition'] = 'PP'
mappings['Punctuation'] = 'PU'
mappings['Object Pronoun'] = 'PO'
mappings['Subject Pronoun'] = 'PS'
mappings['Determinant'] = 'DE'


class WordStructure:
    def __init__(self):
        self.type = ''
        self.gender = ''
        self.person = ''
        self.number = ''
        self.aspect = ''
        self.type = ''
        self.tense = ''
        self.state = ''
        self.radical = ''
        self.annex = ''


class Hypothesis:
    def __init__(self):
        self.gender = []
        self.person = []
        self.number = []
        self.aspect = []
        self.type = []
        self.tense = []
        self.state = []
        self.radical = []
        self.annex = []


general_hypotheses = Hypothesis()
specific_hypotheses = Hypothesis()
unified_hypothesis = Hypothesis()


def check(word, feature):
    if word['type'] == feature.type:
        if feature.gender == '' or feature.gender == word['gender']:
            if feature.person == '' or feature.type == word['person']:
                if feature.number == '' or feature.number == word['number']:
                    if feature.tense == '' or feature.tense == word['tense']:
                        if feature.aspect == '' or feature.aspect == word['aspect']:
                            if feature.type == '' or feature.type == word['type']:
                                if feature.state == '' or feature.state == word['state']:
                                    if feature.radical == '' or feature.radical == word['radical']:
                                        if feature.annex == '' or feature.annex == word['annex']:
                                            return True
    return False


def dfs(index, sentence, pattern):
    if index == len(pattern):
        sentencesToReturn.append(sentence)
        return

    feature = WordStructure()
    feature.type = pattern[index]
    for i in range(0, index):
        if index in unified_hypothesis.gender[i]:
            feature.gender = sentence[i].gender
        if index in unified_hypothesis.person[i]:
            feature.person = sentence[i].person
        if index in unified_hypothesis.number[i]:
            feature.number = sentence[i].number
        if index in unified_hypothesis.tense[i]:
            feature.tense = sentence[i].tense
        if index in unified_hypothesis.aspect[i]:
            feature.aspect = sentence[i].aspect
        if index in unified_hypothesis.radical[i]:
            feature.radical = sentence[i].radical
        if index in unified_hypothesis.annex[i]:
            feature.annex = sentence[i].annex
        if index in unified_hypothesis.state[i]:
            feature.state = sentence[i].state
        if index in unified_hypothesis.type[i]:
            feature.type = sentence[i].type

    for word in dictionary:
        temp_sentence = sentence
        if check(word, feature):
            temp_sentence.append(word)
            dfs(index + 1, temp_sentence, pattern)


def get_word_feature(word, word_type, feature):
    dict_type = dictionary[dictionary['POS'] == word_type]
    if word_type == 'prep' or word_type == 'CONJ':
        word_in_dict = dict_type[dict_type['BASE'] == word]
    else:
        word_in_dict = dict_type[dict_type['FULL'] == word]
    feature_type = word_in_dict[feature]
    return feature_type


def initialize_general_hypotheses(hypothesis_length):
    initial_general = []
    for i in range(1, hypothesis_length):
        matches = [i]
        initial_general.append(matches)
    return initial_general


def initialize_specific_hypotheses(dataset, feature):
    initial_specific = [initialize_general_hypotheses(len(dataset.columns) - 1)]
    for row in dataset:
        if dataset['label'] == 'yes':
            for i in range(0, len(dataset.columns) - 1, -1):
                word = row[i]
                word_type = dataset.columns[i]
                feature_type = get_word_feature(word, word_type, feature)
                for j in range(0, i, -1):
                    word = row[j]
                    word_type = dataset.columns[j]
                    feature_type_to_match = get_word_feature(word, word_type, feature)
                    if feature_type == feature_type_to_match:
                        initial_specific[0][i].append(j)
            break
    return initial_specific


def get_general_hypotheses(pattern):
    hypotheses_file_name = get_filename('_hypotheses', pattern)
    if os.path.exists(hypotheses_file_name):

        total_hypotheses = pd.read_csv(hypotheses_file_name)
        general = total_hypotheses[total_hypotheses['type'] == 'general']

        general_hypotheses.gender = general[general['feature'] == 'gender']
        general_hypotheses.person = general[general['feature'] == 'person']
        general_hypotheses.number = general[general['feature'] == 'number']
        general_hypotheses.aspect = general[general['feature'] == 'aspect']
        general_hypotheses.type = general[general['feature'] == 'type']
        general_hypotheses.tense = general[general['feature'] == 'tense']
        general_hypotheses.state = general[general['feature'] == 'state']
        general_hypotheses.radical = general[general['feature'] == 'radical']
        general_hypotheses.annex = general[general['feature'] == 'annex']

    else:
        hypothesis_size = len(pattern)

        general_hypotheses.gender = initialize_general_hypotheses(hypothesis_size)
        general_hypotheses.person = initialize_general_hypotheses(hypothesis_size)
        general_hypotheses.number = initialize_general_hypotheses(hypothesis_size)
        general_hypotheses.aspect = initialize_general_hypotheses(hypothesis_size)
        general_hypotheses.type = initialize_general_hypotheses(hypothesis_size)
        general_hypotheses.tense = initialize_general_hypotheses(hypothesis_size)
        general_hypotheses.state = initialize_general_hypotheses(hypothesis_size)
        general_hypotheses.radical = initialize_general_hypotheses(hypothesis_size)
        general_hypotheses.annex = initialize_general_hypotheses(hypothesis_size)


def get_specific_hypotheses(pattern):
    hypotheses_file_name = get_filename('_hypotheses', pattern)
    if os.path.exists(hypotheses_file_name):

        total_hypotheses = pd.read_csv(hypotheses_file_name)
        specific = total_hypotheses[total_hypotheses['type'] == 'specific']

        specific_hypotheses.gender = specific[specific['feature'] == 'gender']
        specific_hypotheses.person = specific[specific['feature'] == 'person']
        specific_hypotheses.number = specific[specific['feature'] == 'number']
        specific_hypotheses.aspect = specific[specific['feature'] == 'aspect']
        specific_hypotheses.type = specific[specific['feature'] == 'type']
        specific_hypotheses.tense = specific[specific['feature'] == 'tense']
        specific_hypotheses.state = specific[specific['feature'] == 'state']
        specific_hypotheses.radical = specific[specific['feature'] == 'radical']
        specific_hypotheses.annex = specific[specific['feature'] == 'annex']
    else:
        dataset_filename = get_filename('_dataset', pattern)
        if os.path.exists(dataset_filename):
            dataset = pd.read_csv(dataset_filename)
            specific_hypotheses.gender = initialize_specific_hypotheses(dataset, 'gender')
            specific_hypotheses.person = initialize_specific_hypotheses(dataset, 'person')
            specific_hypotheses.number = initialize_specific_hypotheses(dataset, 'number')
            specific_hypotheses.aspect = initialize_specific_hypotheses(dataset, 'aspect')
            specific_hypotheses.type = initialize_specific_hypotheses(dataset, 'type')
            specific_hypotheses.tense = initialize_specific_hypotheses(dataset, 'tense')
            specific_hypotheses.state = initialize_specific_hypotheses(dataset, 'state')
            specific_hypotheses.radical = initialize_specific_hypotheses(dataset, 'radical')
            specific_hypotheses.annex = initialize_specific_hypotheses(dataset, 'annex')


def get_hypothesis_by_feature(genre, feature):
    if genre == 'specific':
        if feature == 'gender':
            return specific_hypotheses.gender
        elif feature == 'person':
            return specific_hypotheses.person
        elif feature == 'number':
            return specific_hypotheses.number
        elif feature == 'aspect':
            return specific_hypotheses.aspect
        elif feature == 'type':
            return specific_hypotheses.type
        elif feature == 'tense':
            return specific_hypotheses.tense
        elif feature == 'state':
            return specific_hypotheses.state
        elif feature == 'radical':
            return specific_hypotheses.radical
        elif feature == 'annex':
            return specific_hypotheses.annex
    else:
        if feature == 'gender':
            return general_hypotheses.gender
        elif feature == 'person':
            return general_hypotheses.person
        elif feature == 'number':
            return general_hypotheses.number
        elif feature == 'aspect':
            return general_hypotheses.aspect
        elif feature == 'type':
            return general_hypotheses.type
        elif feature == 'tense':
            return general_hypotheses.tense
        elif feature == 'state':
            return general_hypotheses.state
        elif feature == 'radical':
            return general_hypotheses.radical
        elif feature == 'annex':
            return general_hypotheses.annex


def build_hypothesis(sentence, pattern, label, feature):
    get_general_hypotheses(pattern)
    get_specific_hypotheses(pattern)
    specific_feature_hypotheses = get_hypothesis_by_feature('specific', feature)
    general_feature_hypotheses = get_hypothesis_by_feature('general', feature)
    if label == 'yes':
        recorded_changes = []
        for specific in specific_feature_hypotheses:
            for source in range(0, len(pattern), -1):
                word_type = get_word_feature(sentence[source], pattern[source], feature)
                for agreement in specific[source]:
                    word_to_match_type = get_word_feature(sentence[agreement], pattern[agreement], feature)
                    matchy = False
                    for tpe in word_type:
                        for matchtpe in word_to_match_type:
                            if tpe == matchtpe:
                                matchy = True
                                break
                        if matchy:
                            break
                    if not matchy:
                        single_change = specific
                        single_change[source].remove(agreement)
                        recorded_changes.append(single_change)
            if len(recorded_changes) > 0:
                specific_feature_hypotheses.remove(specific)
        for recorded in recorded_changes:
            if recorded not in specific_feature_hypotheses:
                specific_feature_hypotheses.append(recorded)
                new_general = []
                for general in general_feature_hypotheses:
                    conflicting_general_hypothesis = False
                    for index, col in enumerate(general):
                        for agreement in col:
                            if agreement not in recorded[index]:
                                conflicting_general_hypothesis = True
                                break
                        if conflicting_general_hypothesis:
                            break
                    if not conflicting_general_hypothesis:
                        new_general.append(general)
                general_feature_hypotheses = new_general
        for index in range(0, len(pattern)):
            specificity_allowed = 100
            for specific in specific_feature_hypotheses:
                specificity_allowed = min(specificity_allowed, len(specific[index]))
            for specific in specific_feature_hypotheses:
                if len(specific[index]) > specificity_allowed:
                    specific_feature_hypotheses.remove(specific)

    elif label == 'no':
        for source_word in range(0, len(pattern)):
            word_type = get_word_feature(sentence[source_word], pattern[source_word], feature)
            for dest_word in range(source_word + 1, len(pattern)):
                word_to_match_type = get_word_feature(sentence[dest_word], pattern[dest_word], feature)
                matchy = False
                for tpe in word_type:
                    for matchtpe in word_to_match_type:
                        if tpe == matchtpe:
                            matchy = True
                            break
                    if matchy:
                        break
                if not matchy:
                    handled = False
                    for general in general_feature_hypotheses:
                        if dest_word in general[source_word]:
                            handled = True
                            break
                    if not handled:
                        new_general = general_feature_hypotheses[0]
                        for specific in specific_feature_hypotheses:
                            if dest_word in specific[source_word]:
                                new_general[source_word].append(dest_word)
                                handled = True
                                break
                        if handled:
                            del general_feature_hypotheses[0]
                            general_feature_hypotheses.append(new_general)
        for index in range(0, len(pattern)):
            generality_allowed = 0
            for general in general_feature_hypotheses:
                generality_allowed = max(generality_allowed, len(general[index]))
            for general in general_feature_hypotheses:
                if len(general[index]) < generality_allowed:
                    general_feature_hypotheses.remove(general)
    return general_hypotheses, specific_hypotheses


def get_filename(extension, pattern):
    chosen_file_name = ''
    for element in pattern:
        chosen_file_name = chosen_file_name + mappings[element]
    chosen_file_name = chosen_file_name + extension + '.csv'

    return chosen_file_name


def create_dataset(header):
    file_name = get_filename('_dataset', header)
    header.append('label')
    with open(file_name, mode='w') as csv_file:
        fieldnames = header
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    return file_name


def create_hypotheses_file(header):
    file_name = get_filename('_hypotheses', header)
    fieldnames = []
    for pattern in header:
        fieldnames.append(mappings[pattern])
    fieldnames.append('feature')
    fieldnames.append('genre')
    with open(file_name, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    return fieldnames


def append_hypothesis_rows(filename, fieldnames, genre, feature):
    hypotheses_to_save = get_hypothesis_by_feature(genre, feature)
    with open(filename, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_row = dict()
        csv_row['genre'] = genre
        csv_row['feature'] = feature
        for hypothesis in hypotheses_to_save:
            for index, pattern in fieldnames:
                if pattern != 'genre' and pattern != 'feature':
                    csv_row[pattern] = hypothesis[index]
            writer.writerow(csv_row)


def __append_hypo_rows(hypotheses_filename, hypothesis_fieldnames, genre):
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'gender')
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'person')
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'number')
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'tense')
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'aspect')
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'state')
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'radical')
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'annex')
    append_hypothesis_rows(hypotheses_filename, hypothesis_fieldnames, genre, 'type')


@app.route('/saveword', methods=['GET', 'POST'])
@cross_origin(allow_headers=['Content-Type', 'Access-Control-Allow-Origin'])
def add_word_to_dictionary():
    user_input = request.json
    word = user_input['full']
    pos = user_input['pos']
    gender = user_input['gen']
    tense = user_input['ten']
    person = user_input['pers']
    number = user_input['number']
    aspect = user_input['asp']
    tpe = user_input['type']
    state = user_input['st']
    radical = user_input['rad']
    annex = user_input['an']
    word_form = pd.DataFrame()
    word_form['FULL'] = word
    word_form['POS'] = pos
    word_form['GEN'] = gender
    word_form['PERS'] = person
    word_form['NUM'] = number
    word_form['TENSE'] = tense
    word_form['ASPECT'] = aspect
    word_form['TYPE'] = tpe
    word_form['STATE'] = state
    word_form['RADICAL'] = radical
    word_form['ANNEX'] = annex
    dictionary.append(word_form)
    with open('dictionary.csv', mode='a') as csv_file:
        fieldnames = dictionary.columns
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'FULL': word, 'POS': pos, 'GEN': gender, 'PERS': person, 'NUM': number, "TENSE": tense,
                         "ASPECT": aspect, "TYPE": tpe, "STATE": state, "RADICAL": radical, "ANNEX": annex})


@app.route('/generate', methods=['GET', 'POST'])
@cross_origin()
def build_sentence():
    user_input = request.json
    pattern = user_input['pattern']
    num_of_sentences = user_input['number']

    dataset_file_name = get_filename('_dataset', pattern)
    chosen_sentences = []
    if os.path.exists(dataset_file_name):
        dataset = pd.read_csv(dataset_file_name)
        get_general_hypotheses(pattern)
        get_specific_hypotheses(pattern)
        dfs(0, [], pattern)
        for sentence in sentencesToReturn:
            if sentence not in dataset:
                chosen_sentences.append(sentence)
                if len(chosen_sentences) == num_of_sentences:
                    break

    return chosen_sentences


@app.route('/getdict', methods=['GET', 'POST'])
@cross_origin()
def send_dictionary():
    verb = dictionary[dictionary['POS'] == 'V']
    verb = verb['FULL'].tolist()
    cn = dictionary[dictionary['POS'] == 'CN']
    cn = cn['FULL'].tolist()
    pn = dictionary[dictionary['POS'] == 'PN']
    pn = pn['FULL'].tolist()
    adj = dictionary[dictionary['POS'] == 'ADJ']
    adj = adj['FULL'].tolist()
    prep = dictionary[dictionary['POS'] == 'prep']
    prep = prep['BASE'].tolist()
    pron = dictionary[dictionary['POS'] == 'PRON']
    pron = pron['FULL'].tolist()
    conj = dictionary[dictionary['POS'] == 'CONJ']
    conj = conj['BASE'].tolist()
    return jsonify(v=verb, cn=cn, pn=pn, adj=adj, prep=prep, pron=pron, conj=conj)


@app.route('/train', methods=['GET', 'POST'])
@cross_origin()
def get_user_input():
    user_input = request.json
    pattern = user_input['pattern']
    sentence = user_input['sentence']
    dataset_name = get_filename('_dataset', pattern)
    if not os.path.exists(dataset_name):
        if user_input['label'] == 'no':
            return 'This is a new pattern, please start with a positive example'
        dataset_name = create_dataset(pattern)

    general_hypotheses.gender, specific_hypotheses.gender = build_hypothesis(sentence, pattern, user_input['label'],
                                                                             'gender')
    general_hypotheses.person, specific_hypotheses.person = build_hypothesis(sentence, pattern, user_input['label'],
                                                                             'person')
    general_hypotheses.number, specific_hypotheses.number = build_hypothesis(sentence, pattern, user_input['label'],
                                                                             'number')
    general_hypotheses.tense, specific_hypotheses.tense = build_hypothesis(sentence, pattern, user_input['label'],
                                                                           'tense')
    general_hypotheses.aspect, specific_hypotheses.aspect = build_hypothesis(sentence, pattern, user_input['label'],
                                                                             'aspect')
    general_hypotheses.type, specific_hypotheses.type = build_hypothesis(sentence, pattern, user_input['label'],
                                                                         'type')
    general_hypotheses.radical, specific_hypotheses.radical = build_hypothesis(sentence, pattern, user_input['label'],
                                                                               'radical')
    general_hypotheses.state, specific_hypotheses.state = build_hypothesis(sentence, pattern, user_input['label'],
                                                                           'state')
    general_hypotheses.annex, specific_hypotheses.annex = build_hypothesis(sentence, pattern, user_input['label'],
                                                                           'annex')

    hypotheses_filename = get_filename('_hypotheses', pattern)
    hypothesis_fieldnames = create_hypotheses_file(pattern)

    __append_hypo_rows(hypotheses_filename, hypothesis_fieldnames, 'general')
    __append_hypo_rows(hypotheses_filename, hypothesis_fieldnames, 'specific')
    with open(dataset_name, mode='a') as csv_file:
        fieldnames = pattern
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_row = dict()
        for index, element in enumerate(pattern):
            csv_row[element] = sentence[index]
        csv_row['label'] = user_input['yes']
        writer.writerow(csv_row)
    return 'Tudo Bem'


if __name__ == '__main__':
    app.run(debug=True)
