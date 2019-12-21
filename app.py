from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask import jsonify
import pandas as pd
import numpy as np
import random
import os.path

app = Flask(__name__)
CORS(app, resources=r'*')
dictionary = pd.read_excel('lexicon.xlsx', sheet_name='V')
dictionary = dictionary.replace(np.nan, '', regex=True)
sentencesToReturn = []
to_duplicate = ['Common Noun', 'Proper Noun', 'Verb', 'Adjective', 'Preposition', 'Pronoun', 'Conjunction',
                'Punctuation']
mappings = dict()
mappings['Common Noun'] = 'CN'
mappings['Proper Noun'] = 'PN'
mappings['Verb'] = 'V'
mappings['Adjective'] = 'ADJ'
mappings['Preposition'] = 'prep'
mappings['Punctuation'] = 'PU'
mappings['Pronoun'] = 'PRON'
mappings['Conjunction'] = 'CONJ'
mappings['None'] = ''
mappings[''] = ''
mappings['M'] = 'm'
mappings['F'] = 'f'
mappings['Possessive'] = 'poss'
mappings['Person Free'] = 'persfree'
mappings['1st Person'] = 1
mappings['2nd Person'] = 2
mappings['3rd Person'] = 3
mappings['Singular'] = 's'
mappings['Plural'] = 'p'
mappings['Aorist'] = 'aorist'
mappings['Imperfect'] = 'imperf'
mappings['Perfect Positive'] = 'perfpos'
mappings['Perfect Negative'] = 'perfneg'
mappings['Annex'] = 'annex'
mappings['Free'] = 'free'
mappings['Yes'] = 'y'
mappings['No'] = 'n'
mappings['Both'] = 'd'
mappings['Past'] = 1
mappings['Present'] = 2
mappings['Future'] = 3
mappings['gender'] = 'GEN'
mappings['type'] = 'TYPE'
mappings['pos'] = 'POS'
mappings['person'] = 'PERS'
mappings['tense'] = 'TENSE'
mappings['number'] = 'NUM'
mappings['state'] = 'STATE'
mappings['radical'] = 'RADICAL'
mappings['annex'] = 'ANNEX'
mappings['aspect'] = 'ASPECT'
for part_of_speech in to_duplicate:
    for k in range(1, 10):
        dup = part_of_speech + str(k)
        mappings[dup] = mappings[part_of_speech]


class WordStructure:
    def __init__(self):
        self.pos = ''
        self.gender = []
        self.person = []
        self.number = []
        self.aspect = []
        self.type = []
        self.tense = []
        self.state = []
        self.annex = []


class Hypothesis:
    def __init__(self):
        self.gender = []
        self.person = []
        self.number = []
        self.aspect = []
        self.type = []
        self.tense = []
        self.state = []
        self.annex = []


general_hypotheses = Hypothesis()
specific_hypotheses = Hypothesis()
unified_specific_hypothesis = Hypothesis()
unified_general_hypothesis = Hypothesis()


def check(word, feature):
    if word['POS'] == mappings[feature.pos]:
        if feature.gender == [] or word['GEN'] in feature.gender:
            if feature.person == [] or word['PERS'] in feature.person:
                if feature.number == [] or word['NUM'] in feature.number:
                    if feature.tense == [] or word['TENSE'] in feature.tense:
                        if feature.aspect == [] or word['ASPECT'] in feature.aspect:
                            if feature.type == [] or word['TYPE'] in feature.type:
                                if feature.state == [] or word['STATE'] in feature.state:
                                    if feature.annex == [] or word['ANNEX'] in feature.annex:
                                        return True
    return False


def get_word_feature(word, word_type, feature):
    dict_type = dictionary[dictionary['POS'] == mappings[word_type]].copy()
    if mappings[word_type] == 'prep' or mappings[word_type] == 'CONJ':
        word_in_dict = dict_type[dict_type['BASE'] == word]
    else:
        word_in_dict = dict_type[dict_type['FULL'] == word]
    feature_type = word_in_dict[mappings[feature]].tolist()
    return feature_type


def dfs(index, sentence, pattern, number_needed, tpe):
    if len(sentencesToReturn) == (10 * number_needed):
        return

    if index == len(pattern):
        sentencesToReturn.append(sentence)
        return

    feature = WordStructure()
    feature.pos = pattern[index]
    if type == 'specific':
        usable_hypotheses = unified_specific_hypothesis
    else:
        usable_hypotheses = unified_general_hypothesis
    for i in range(0, index):
        if index in usable_hypotheses.gender[i]:
            feature.gender = get_word_feature(sentence[i], pattern[i], 'gender')
        if index in usable_hypotheses.person[i]:
            feature.person = get_word_feature(sentence[i], pattern[i], 'person')
        if index in usable_hypotheses.number[i]:
            feature.number = get_word_feature(sentence[i], pattern[i], 'number')
        if index in usable_hypotheses.tense[i]:
            feature.tense = get_word_feature(sentence[i], pattern[i], 'tense')
        if index in usable_hypotheses.aspect[i]:
            feature.aspect = get_word_feature(sentence[i], pattern[i], 'aspect')
        if index in usable_hypotheses.annex[i]:
            feature.annex = get_word_feature(sentence[i], pattern[i], 'annex')
        if index in usable_hypotheses.state[i]:
            feature.state = get_word_feature(sentence[i], pattern[i], 'state')
        if index in usable_hypotheses.type[i]:
            feature.type = get_word_feature(sentence[i], pattern[i], 'type')

    global dictionary
    dictionary = dictionary.sample(frac=1)
    for i, row in dictionary.iterrows():
        temp_sentence = sentence[:]
        if check(row, feature):
            if row['POS'] == 'prep' or row['POS'] == 'CONJ':
                temp_sentence.append(row['BASE'])
            else:
                temp_sentence.append(row['FULL'])
            dfs(index + 1, temp_sentence, pattern, number_needed, tpe)


def initialize_general_hypotheses(hypothesis_length):
    initial_general = []
    for i in range(0, hypothesis_length):
        matches = [i]
        initial_general.append(matches)
    return initial_general


def initialize_specific_hypotheses(dataset, feature):
    initial_specific = [initialize_general_hypotheses(len(dataset.columns) - 1)]
    cols = dataset.columns[:]
    for index, row in dataset.iterrows():
        if row['label'] == 'yes':
            for i in range(0, len(cols) - 1):
                word = row[cols[i]]
                word_type = cols[i]
                feature_type = get_word_feature(word, word_type, feature)
                for j in range(0, i):
                    word = row[cols[j]]
                    word_type = cols[j]
                    feature_type_to_match = get_word_feature(word, word_type, feature)
                    matchy = False
                    for feat in feature_type:
                        if feat in feature_type_to_match:
                            matchy = True
                            break
                    if matchy:
                        initial_specific[0][j].append(i)
            break
    return initial_specific


def populate_from_excel(df, hypos, cols):
    temp_hypo = []
    for index, row in df.iterrows():
        for col in cols:
            if col != 'genre' and col != 'feature':
                list_hypo = []
                for c in row[col]:
                    if '0' <= c <= '9':
                        list_hypo.append(int(c))
                temp_hypo.append(list_hypo)
        hypos.append(temp_hypo)


def get_general_hypotheses(pattern):
    hypotheses_file_name = get_filename('_hypotheses', pattern)
    if os.path.exists(hypotheses_file_name):
        total_hypotheses = pd.read_excel(hypotheses_file_name, sheet_name='Sheet1')
        cols = total_hypotheses.columns
        general = total_hypotheses[total_hypotheses['genre'] == 'general']
        populate_from_excel(general[general['feature'] == 'gender'], general_hypotheses.gender, cols)
        populate_from_excel(general[general['feature'] == 'person'], general_hypotheses.person, cols)
        populate_from_excel(general[general['feature'] == 'number'], general_hypotheses.number, cols)
        populate_from_excel(general[general['feature'] == 'tense'], general_hypotheses.tense, cols)
        populate_from_excel(general[general['feature'] == 'aspect'], general_hypotheses.aspect, cols)
        populate_from_excel(general[general['feature'] == 'type'], general_hypotheses.type, cols)
        populate_from_excel(general[general['feature'] == 'state'], general_hypotheses.state, cols)
        populate_from_excel(general[general['feature'] == 'annex'], general_hypotheses.annex, cols)

    else:
        hypothesis_size = len(pattern)

        general_hypotheses.gender = [initialize_general_hypotheses(hypothesis_size)]
        general_hypotheses.person = [initialize_general_hypotheses(hypothesis_size)]
        general_hypotheses.number = [initialize_general_hypotheses(hypothesis_size)]
        general_hypotheses.aspect = [initialize_general_hypotheses(hypothesis_size)]
        general_hypotheses.type = [initialize_general_hypotheses(hypothesis_size)]
        general_hypotheses.tense = [initialize_general_hypotheses(hypothesis_size)]
        general_hypotheses.state = [initialize_general_hypotheses(hypothesis_size)]
        general_hypotheses.annex = [initialize_general_hypotheses(hypothesis_size)]


def get_specific_hypotheses(pattern, sentence, label):
    hypotheses_file_name = get_filename('_hypotheses', pattern)
    if os.path.exists(hypotheses_file_name):

        total_hypotheses = pd.read_excel(hypotheses_file_name, sheet_name='Sheet1')
        specific = total_hypotheses[total_hypotheses['genre'] == 'specific']
        cols = total_hypotheses.columns
        if len(specific_hypotheses.gender) == 0:
            populate_from_excel(specific[specific['feature'] == 'gender'], specific_hypotheses.gender, cols)
            populate_from_excel(specific[specific['feature'] == 'person'], specific_hypotheses.person, cols)
            populate_from_excel(specific[specific['feature'] == 'number'], specific_hypotheses.number, cols)
            populate_from_excel(specific[specific['feature'] == 'aspect'], specific_hypotheses.aspect, cols)
            populate_from_excel(specific[specific['feature'] == 'type'], specific_hypotheses.type, cols)
            populate_from_excel(specific[specific['feature'] == 'tense'], specific_hypotheses.tense, cols)
            populate_from_excel(specific[specific['feature'] == 'state'], specific_hypotheses.state, cols)
            populate_from_excel(specific[specific['feature'] == 'annex'], specific_hypotheses.annex, cols)
    else:
        name_pattern = pattern[:]
        dataset_filename = get_filename('_dataset', name_pattern)
        if not os.path.exists(dataset_filename):
            dataset = pd.DataFrame(columns=pattern)
            dict_for_ds = dict()
            for index, element in enumerate(pattern):
                dict_for_ds[element] = sentence[index]
            dict_for_ds['label'] = label
            dataset = dataset.append(dict_for_ds, ignore_index=True)
        else:
            dataset = pd.read_excel(dataset_filename, sheet_name='Sheet1')
        if len(specific_hypotheses.gender) == 0:
            specific_hypotheses.gender = initialize_specific_hypotheses(dataset, 'gender')
            specific_hypotheses.person = initialize_specific_hypotheses(dataset, 'person')
            specific_hypotheses.number = initialize_specific_hypotheses(dataset, 'number')
            specific_hypotheses.aspect = initialize_specific_hypotheses(dataset, 'aspect')
            specific_hypotheses.type = initialize_specific_hypotheses(dataset, 'type')
            specific_hypotheses.tense = initialize_specific_hypotheses(dataset, 'tense')
            specific_hypotheses.state = initialize_specific_hypotheses(dataset, 'state')
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
        elif feature == 'annex':
            return general_hypotheses.annex


def build_hypothesis(sentence, pattern, label, feature):
    name_pattern = pattern[:]
    if len(general_hypotheses.gender) == 0:
        get_general_hypotheses(name_pattern)
    name_pattern = pattern[:]
    if len(specific_hypotheses.gender) == 0:
        get_specific_hypotheses(name_pattern, sentence, label)
    specific_feature_hypotheses = get_hypothesis_by_feature('specific', feature)
    general_feature_hypotheses = get_hypothesis_by_feature('general', feature)
    if label == 'yes':
        recorded_changes = []
        for specific in specific_feature_hypotheses:
            for source in range(0, len(pattern)):
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


def get_filename(extension, pattern):
    chosen_file_name = ''
    for element in pattern:
        chosen_file_name = chosen_file_name + mappings[element]
    chosen_file_name = chosen_file_name + extension + '.xlsx'

    return chosen_file_name


def create_dataset(header):
    file_name = get_filename('_dataset', header)
    header.append('label')
    dataset = pd.DataFrame(columns=header)
    dataset.to_excel(file_name, startrow=0, sheet_name='Sheet1', index=False)
    return file_name


def create_hypotheses_file(header):
    file_name = get_filename('_hypotheses', header)
    fieldnames = []
    for pattern in header:
        fieldnames.append(mappings[pattern])
    fieldnames.append('feature')
    fieldnames.append('genre')
    dataset = pd.DataFrame(columns=fieldnames)
    dataset.to_excel(file_name, startrow=0, sheet_name='Sheet1', index=False)
    return fieldnames


def unify_hypotheses(hypos, unified_hypos, index):
    most_specific = 0
    chosen = []
    for specific in hypos:
        if len(specific[index]) < most_specific:
            most_specific = len(specific[index])
            chosen = specific[index]
    unified_hypos.append(chosen)


def unify_general(hypos, unified_hypos, index):
    most_general = 0
    chosen = []
    for general in hypos:
        if len(general[index]) > most_general:
            most_general = len(general[index])
            chosen = general[index]
    unified_hypos.append(chosen)


@app.route('/saveword', methods=['GET', 'POST'])
@cross_origin(allow_headers=['Content-Type', 'Access-Control-Allow-Origin'])
def add_word_to_dictionary():
    global dictionary
    user_input = request.json
    word = user_input['full']
    radical = user_input['rad']
    pos = mappings[user_input['pos']]
    gender = mappings[user_input['gen']]
    tense = mappings[user_input['ten']]
    person = mappings[user_input['pers']]
    number = mappings[user_input['number']]
    aspect = mappings[user_input['asp']]
    tpe = mappings[user_input['type']]
    state = mappings[user_input['st']]
    annex = mappings[user_input['an']]

    if pos == 'prep' or pos == 'conj':
        dictionary = dictionary.append(
            {'BASE': word, 'POS': pos, 'GEN': gender, 'PERS': person, 'NUM': number, 'TENSE': tense,
             'ASPECT': aspect, 'TYPE': tpe, 'STATE': state, 'RADICAL': radical, 'ANNEX': annex},
            ignore_index=True)
    else:
        dictionary = dictionary.append(
            {'FULL': word, 'POS': pos, 'GEN': gender, 'PERS': person, 'NUM': number, 'TENSE': tense,
             'ASPECT': aspect, 'TYPE': tpe, 'STATE': state, 'RADICAL': radical, 'ANNEX': annex},
            ignore_index=True)
    dictionary.to_excel("lexicon.xlsx", startrow=0, sheet_name='V', index=False)
    return jsonify(resp='All Good Ma Main')


@app.route('/generate', methods=['GET', 'POST'])
@cross_origin()
def build_sentence():
    user_input = request.json
    pattern = user_input['pattern']
    num_of_sentences = user_input['number']
    for i in range(0, len(pattern)):
        last_dig = ''
        idx = -1
        for j in range(0, i):
            if pattern[i] in pattern[j]:
                idx = j
                last_dig = str(pattern[j][len(pattern[j]) - 1])
        if idx != -1:
            if '1' <= last_dig <= '9':
                pattern[i] += chr(ord(last_dig) + 1)
            else:
                pattern[i] += '1'
    name_pattern = pattern[:]
    dataset_file_name = get_filename('_dataset', name_pattern)
    name_pattern = pattern[:]
    hypotheses_file_name = get_filename('_hypotheses', name_pattern)
    chosen_sentences = []
    if os.path.exists(dataset_file_name) and os.path.exists(hypotheses_file_name):
        dataset = pd.read_excel(dataset_file_name, sheet_name='Sheet1')
        name_pattern = pattern[:]
        get_specific_hypotheses(name_pattern, [], '')
        name_pattern = pattern[:]
        get_general_hypotheses(name_pattern)
        for i in range(0, len(pattern)):
            unify_hypotheses(specific_hypotheses.gender, unified_specific_hypothesis.gender, i)
            unify_hypotheses(specific_hypotheses.person, unified_specific_hypothesis.person, i)
            unify_hypotheses(specific_hypotheses.number, unified_specific_hypothesis.number, i)
            unify_hypotheses(specific_hypotheses.tense, unified_specific_hypothesis.tense, i)
            unify_hypotheses(specific_hypotheses.aspect, unified_specific_hypothesis.aspect, i)
            unify_hypotheses(specific_hypotheses.state, unified_specific_hypothesis.state, i)
            unify_hypotheses(specific_hypotheses.annex, unified_specific_hypothesis.annex, i)
            unify_hypotheses(specific_hypotheses.type, unified_specific_hypothesis.type, i)
            unify_general(general_hypotheses.gender, unified_general_hypothesis.gender, i)
            unify_general(general_hypotheses.person, unified_general_hypothesis.person, i)
            unify_general(general_hypotheses.number, unified_general_hypothesis.number, i)
            unify_general(general_hypotheses.tense, unified_general_hypothesis.tense, i)
            unify_general(general_hypotheses.aspect, unified_general_hypothesis.aspect, i)
            unify_general(general_hypotheses.state, unified_general_hypothesis.state, i)
            unify_general(general_hypotheses.annex, unified_general_hypothesis.annex, i)
            unify_general(general_hypotheses.type, unified_general_hypothesis.type, i)

        name_pattern = pattern[:]
        global sentencesToReturn
        sentencesToReturn = []
        dfs(0, [], name_pattern, num_of_sentences/2, 'specific')
        if (num_of_sentences - (num_of_sentences/2)) > 0:
            dfs(0, [], name_pattern, num_of_sentences - (num_of_sentences/2), 'general')
        random.shuffle(sentencesToReturn)
        random.shuffle(sentencesToReturn)
        random.shuffle(sentencesToReturn)
        for index, sentence in enumerate(sentencesToReturn):
            if len(chosen_sentences) == num_of_sentences:
                break
            if (num_of_sentences - len(chosen_sentences)) == (len(sentencesToReturn) - index):
                for i in range(index, len(sentencesToReturn)):
                    chosen_sentences.append(sentencesToReturn[i])
            else:
                statement = True
                for i, element in enumerate(pattern):
                    statement = statement & (dataset[element] == sentence[i])
                statement = statement.any()
                if not statement:
                    chosen_sentences.append(sentence)
                    if len(chosen_sentences) == num_of_sentences:
                        break
        response = jsonify(response='All Good Ma Main', sentences=chosen_sentences)
    else:
        response = jsonify(response='Not Trained Yet', sentences=[])

    return response


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


def build_dict(df, hypos, genre, feature, pattern):
    for hypo in hypos:
        excel_hypo_row = dict()
        for index, element in enumerate(pattern):
            excel_hypo_row[element] = hypo[index]
        excel_hypo_row['feature'] = feature
        excel_hypo_row['genre'] = genre
        df = df.append(excel_hypo_row, ignore_index=True)
    return df


def remove_duplicates(hypo):
    return [hypo[x] for x in range(len(hypo)) if not (hypo[x] in hypo[:x])]


@app.route('/train', methods=['GET', 'POST'])
@cross_origin()
def get_user_input():
    user_input = request.json
    pattern = user_input['pattern']
    sentence = user_input['sentence']
    label = user_input['label']
    for i in range(0, len(pattern)):
        last_dig = ''
        idx = -1
        for j in range(0, i):
            if pattern[i] in pattern[j]:
                idx = j
                last_dig = str(pattern[j][len(pattern[j]) - 1])
        if idx != -1:
            if '1' <= last_dig <= '9':
                pattern[i] += chr(ord(last_dig) + 1)
            else:
                pattern[i] += '1'
    name_pattern = pattern[:]
    hypotheses_filename = get_filename('_hypotheses', name_pattern)
    name_pattern = pattern[:]
    dataset_name = get_filename('_dataset', name_pattern)
    if label == 'no' and not os.path.exists(dataset_name) and not os.path.exists(hypotheses_filename):
        return jsonify(resp='Not trained yet')
    name_pattern = pattern[:]
    name_pattern.append('feature')
    name_pattern.append('genre')
    hypotheses_to_save = pd.DataFrame(columns=name_pattern)
    name_pattern = pattern[:]
    build_hypothesis(sentence, name_pattern, user_input['label'], 'gender')
    remove_duplicates(general_hypotheses.gender)
    general_hypotheses.gender = remove_duplicates(general_hypotheses.gender)
    specific_hypotheses.gender = remove_duplicates(specific_hypotheses.gender)
    hypotheses_to_save = build_dict(hypotheses_to_save, general_hypotheses.gender, 'general', 'gender', name_pattern)
    hypotheses_to_save = build_dict(hypotheses_to_save, specific_hypotheses.gender, 'specific', 'gender', name_pattern)
    name_pattern = pattern[:]
    build_hypothesis(sentence, name_pattern, user_input['label'], 'person')
    general_hypotheses.person = remove_duplicates(general_hypotheses.person)
    specific_hypotheses.person = remove_duplicates(specific_hypotheses.person)
    hypotheses_to_save = build_dict(hypotheses_to_save, general_hypotheses.person, 'general', 'person', name_pattern)
    hypotheses_to_save = build_dict(hypotheses_to_save, specific_hypotheses.person, 'specific', 'person', name_pattern)
    name_pattern = pattern[:]
    build_hypothesis(sentence, name_pattern, user_input['label'], 'number')
    general_hypotheses.number = remove_duplicates(general_hypotheses.number)
    specific_hypotheses.number = remove_duplicates(specific_hypotheses.number)
    hypotheses_to_save = build_dict(hypotheses_to_save, general_hypotheses.number, 'general', 'number', name_pattern)
    hypotheses_to_save = build_dict(hypotheses_to_save, specific_hypotheses.number, 'specific', 'number', name_pattern)
    name_pattern = pattern[:]
    build_hypothesis(sentence, name_pattern, user_input['label'], 'tense')
    general_hypotheses.tense = remove_duplicates(general_hypotheses.tense)
    specific_hypotheses.tense = remove_duplicates(specific_hypotheses.tense)
    hypotheses_to_save = build_dict(hypotheses_to_save, general_hypotheses.tense, 'general', 'tense', name_pattern)
    hypotheses_to_save = build_dict(hypotheses_to_save, specific_hypotheses.tense, 'specific', 'tense', name_pattern)
    name_pattern = pattern[:]
    build_hypothesis(sentence, name_pattern, user_input['label'], 'aspect')
    general_hypotheses.aspect = remove_duplicates(general_hypotheses.aspect)
    specific_hypotheses.aspect = remove_duplicates(specific_hypotheses.aspect)
    hypotheses_to_save = build_dict(hypotheses_to_save, general_hypotheses.aspect, 'general', 'aspect', name_pattern)
    hypotheses_to_save = build_dict(hypotheses_to_save, specific_hypotheses.aspect, 'specific', 'aspect', name_pattern)
    name_pattern = pattern[:]
    build_hypothesis(sentence, name_pattern, user_input['label'], 'type')
    general_hypotheses.type = remove_duplicates(general_hypotheses.type)
    specific_hypotheses.type = remove_duplicates(specific_hypotheses.type)
    hypotheses_to_save = build_dict(hypotheses_to_save, general_hypotheses.type, 'general', 'type', name_pattern)
    hypotheses_to_save = build_dict(hypotheses_to_save, specific_hypotheses.type, 'specific', 'type', name_pattern)
    name_pattern = pattern[:]
    build_hypothesis(sentence, name_pattern, user_input['label'], 'state')
    general_hypotheses.state = remove_duplicates(general_hypotheses.state)
    specific_hypotheses.state = remove_duplicates(specific_hypotheses.state)
    hypotheses_to_save = build_dict(hypotheses_to_save, general_hypotheses.state, 'general', 'state', name_pattern)
    hypotheses_to_save = build_dict(hypotheses_to_save, specific_hypotheses.state, 'specific', 'state', name_pattern)
    name_pattern = pattern[:]
    build_hypothesis(sentence, name_pattern, user_input['label'], 'annex')
    general_hypotheses.annex = remove_duplicates(general_hypotheses.annex)
    specific_hypotheses.annex = remove_duplicates(specific_hypotheses.annex)
    hypotheses_to_save = build_dict(hypotheses_to_save, general_hypotheses.annex, 'general', 'annex', name_pattern)
    hypotheses_to_save = build_dict(hypotheses_to_save, specific_hypotheses.annex, 'specific', 'annex', name_pattern)
    name_pattern = pattern[:]
    hypotheses_filename = get_filename('_hypotheses', name_pattern)
    hypotheses_to_save.to_excel(hypotheses_filename, startrow=0, sheet_name='Sheet1', index=False)
    name_pattern = pattern[:]
    dataset_name = get_filename('_dataset', name_pattern)
    name_pattern = pattern[:]
    if not os.path.exists(dataset_name):
        if user_input['label'] == 'no':
            return 'This is a new pattern, please start with a positive example'
        dataset_name = create_dataset(name_pattern)
    dataset = pd.read_excel(dataset_name, sheet_name='Sheet1')
    dict_for_excel = dict()
    for index, element in enumerate(pattern):
        dict_for_excel[element] = sentence[index]
    dict_for_excel['label'] = label
    dataset = dataset.append(dict_for_excel, ignore_index=True)
    dataset.to_excel(dataset_name, startrow=0, sheet_name='Sheet1', index=False)
    return jsonify(resp='Tudo Bem')


if __name__ == '__main__':
    app.run(debug=True)
