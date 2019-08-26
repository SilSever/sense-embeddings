import os
import re
from collections import defaultdict

from lxml.etree import iterparse
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from config import TOM, EUROSENSE, SEW
from config import TOM_SENTENCES, EURO_SENTENCES, SEW_SENTENCES
from text_utils import build_bn2wns, build_wns2bn, write_file

"""
:author: Silvio Severino
"""


def is_valid(lemma, bnsyn, bn2wns):
    """
    It checks if an lemma is valid i.e.:
        - whether it is in the map Babelnet -> Wordnet
        - whether it corresponds to its wordnet synsets
    Note: now this method is not use because for all datasets used,
          this check doesn't improve the score
    :param lemma: lemma to check
    :param bnsyn: Babelnet synset of given lemma
    :param bn2wns: map Babelnet syn to Wordnet syn
    :return: True if it is valid, False otherwise
    """
    res = False

    if not bn2wns.get(bnsyn):
        return res

    # for each wordnet synset associated
    for wns in bn2wns[bnsyn]:

        synset = wn.synset_from_pos_and_offset(wns[-1], int(wns[:-1]))
        lemma_name = synset.lemma_names()
        splitted = lemma.lower().split()

        # if lemma is in the query results
        res = "_".join(splitted) in lemma_name or any([w in lemma_name for w in splitted])

    return res


def parser_eurosense(bn2wns, hc=0, path=EUROSENSE):
    """
    It parses the eurosense dataset, both the high coverage then the high precision.
    In particular, for each sentence (text tag), it retrieves all annotations (anchor, lemma and Babelnet syn)
    and replaces the annotations in just one sentence.
    Note:
        For more details see the corresponding paper and README.md
        at http://lcl.uniroma1.it/eurosense/
    :param bn2wns: map Babelnet synset to Wordnet synset
    :param hc: if 0 it does the function for high coverage
               if 1 it does the function for high precision
    :param path: dataset path, by default eurosense high precision path
    :return: a List of parsed sentences
    """
    tmp_sentence = ""
    out_sentences = []
    annotations = []

    # loads only the given tags, faster than looking for they
    context = iterparse(
        path, events=("end",), tag=["text", "annotation", "annotations"]
    )
    for event, elem in tqdm(context, desc='Parsing Eurosense'):

        # I have to give the sentence
        if elem.tag == "text" and elem.attrib["lang"] == "en":
            annotations = []
            tmp_sentence = elem.text

        # I have to give the id to replace
        elif (
            elem.tag == "annotation"
            and elem.attrib["lang"] == "en"
            and bn2wns.get(elem.text)):
            # and is_valid(elem.attrib["lemma"], elem.text, bn2wns)

            annotations.append(
                {
                    "anchor": elem.attrib["anchor"],
                    "lemma": elem.attrib["lemma"],
                    "score": elem.attrib["coherenceScore"],
                    "syn": elem.text,
                }
            )

        elif elem.tag == "annotations" and event == "end" and tmp_sentence is not None:

            # if hc is 0, it sorts by coverance score, by number of lemma words otherwise
            annotations = sorted(
                annotations,
                key=lambda k: k["score"] if hc == 0 else len(k["anchor"].split(" ")),
                reverse=True,
            )

            # do the replaces
            for annotation in annotations:
                tmp_sentence = tmp_sentence.replace(
                    " " + annotation["anchor"] + " ",
                    " " + "_".join(re.split("-| ", annotation["lemma"])) + "_" + annotation["syn"] + " ")

            out_sentences.append(tmp_sentence)

        # fast iter, clear the memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return out_sentences


def parser_eurosense_line_by_line(bn2wns, path=EUROSENSE):
    """
    It parses the eurosense dataset.
    In particular, for each sentence (text tag), it retrieves all annotations (anchor, lemma and Babelnet syn)
    and replaces just one annotation in just one sentence.
        So for ex, if a sentence has 5 annotations, it returns 5 sentences (one for each one annotation).
    Note:
        - This method is not used because in all experiment doesn't improve the score.
        - For more details see the corresponding paper and README.md
          at http://lcl.uniroma1.it/eurosense/
    :param bn2wns: map Babelnet synset to Wordnet synset
    :param path: dataset path, by default eurosense high precision path
    :return: a List of parsed sentences
    """
    tmp_sentence = ""
    tmp_dict = {}
    out_sentences = []
    for event, elem in iterparse(path):

        # I have to give the sentence
        if elem.tag == "text" and elem.attrib["lang"] == "en":
            tmp_dict = {}
            tmp_sentence = elem.text

        # I have to give the id to replace
        elif elem.tag == "annotation" and elem.attrib["lang"] == "en":
            if elem.text in bn2wns:

                anchor = elem.attrib["anchor"]
                if anchor in tmp_sentence:
                    if anchor not in tmp_dict:
                        tmp_dict[anchor] = tmp_sentence.count(anchor)

                    to_rep = tmp_dict[anchor]

                    out = tmp_sentence.split(anchor, to_rep - 1)
                    out[to_rep - 1] = out[to_rep - 1].replace(
                        " " + anchor + " ",
                        " " + "_".join(elem.attrib["lemma"].split(" ")) + "_" + elem.text + " ",
                        1)
                    out_sentences.append(anchor.join(out))

                    if to_rep != 1:
                        tmp_dict[anchor] -= 1

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return out_sentences


def parser_tom(wns2bn, path=TOM):
    """
    This method parses Tom dataset.
    For each sentence (divided in context sx tag, head tag and context dx tag)
    it retrieves all instances and build a map
    having sentence -> list of set of instances to replace.
    After that for each sentence in the map it replaces the annotations in just one sentence.
    Note:
        For more details see the corresponding paper and README.md
        at http://trainomatic.org/
    :param wns2bn: map Wordnet synset to Babelnet synset
    :param path: dataset path, by default Tom dataset
    :return: a List of parsed sentences
    """
    sentences_set = defaultdict(list)
    sentences = []
    sentence_sx, sentence_dx, to_sub, bn_id = '', '', '', ''
    lemmatizer = WordNetLemmatizer()

    # loads only the given tags, faster than looking for they
    context = iterparse(
        path, events=('end', 'start'), tag=['corpus', 'head', 'answer', 'context', 'instance']
    )
    for event, elem in tqdm(context, desc='Parsing TOM'):

        if elem.tag == 'answer':
            bn_id = wns2bn[elem.attrib['senseId'][3:]]

        if elem.tag == 'context':
            sentence_sx = elem.text

        if elem.tag == 'head':
            to_sub = elem.text
            sentence_dx = elem.tail

        if event == 'end' and elem.tag == 'instance':

            # lemmatize the anchor instance using pos of Wordnet synset
            lemma_bn = lemmatizer.lemmatize(to_sub, pos=bn_id[-1])

            # check whether all piece of sentence is not None
            if all((piece is not None) for piece in [lemma_bn, sentence_sx, sentence_dx, to_sub]):
                sentence_ori = sentence_sx + to_sub + sentence_dx

                sentences_set[sentence_ori].append(
                    {'id': bn_id, 'lemma': lemma_bn, 'anchor': to_sub})

        if event == 'end' and elem.tag == 'corpus':
            for sentence in tqdm(sentences_set, desc='Parsed sentences'):
                tmp = sentence

                for sen_dict in sentences_set[sentence]:
                    tmp = tmp.replace(
                        " " + sen_dict["anchor"] + " ",
                        " " + "_".join(re.split("-| ", sen_dict["lemma"])) + "_" + sen_dict["id"] + " ")
                sentences.append(tmp)

    return sentences


def parser_sew(bn2wns, path=SEW):
    """
    This method parses Sew datasets.
    For each sentence (text tag) it retrieves all annotation (mention and babelNetID)
    and replaces the annotations in just one sentence.
    Note:
        - it iterates through Sew folders
        - for more details see the corresponding paper and README.md
          at http://lcl.uniroma1.it/sew/
    :param bn2wns: map Babelnet synset to Wordnet synset
    :param path: dataset path, by default SEW
    :return: a List of parsed sentences
    """
    parsed_sentences = []
    lemmatizer = WordNetLemmatizer()

    # iterate through Sew folders
    for root, dirs, files in tqdm(os.walk(path), desc='Parsing SEW'):
        for name in files:

            sentence, bnsyn, anchor, lang = '', '', '', ''

            # loads only the given tags, faster than looking for they
            context = iterparse(os.path.join(root, name),
                                events=('start', 'end'),
                                tag=('wikiArticle', 'text', 'annotations',
                                     'annotation', 'babelNetID', 'mention'),
                                encoding='utf-8',
                                recover=True)

            for event, elem in context:
                if elem.tag == 'wikiArticle':
                    lang = elem.attrib['language']

                if lang == 'EN':
                    if elem.tag == 'text':
                        sentence = elem.text

                    if elem.tag == 'babelNetID':
                        bnsyn = elem.text

                    if elem.tag == 'mention':
                        anchor = elem.text

                    if event == 'end' and elem.tag == 'annotation':
                        # check if the Babelnet syn is in the map and whether all piece of sentence is not None
                        if bn2wns.get(bnsyn) and all((piece is not None) for piece in [sentence, bnsyn, anchor]):

                            # lemmatize the anchor instance using pos of Wordnet synset
                            lemma = lemmatizer.lemmatize(anchor, pos=bnsyn[-1])
                            sentence = sentence.replace(' '+anchor+' ',
                                                        " " + "_".join(re.split("-| ", lemma)) + "_" + bnsyn + " ")

                    if event == 'end' and elem.tag == 'annotations':
                        parsed_sentences.append(sentence)

    return parsed_sentences


def lemmer(path=SEW_SENTENCES):

    lemmatizer = WordNetLemmatizer()
    ctx_in = open(path)
    ctx_uout = open("sew_new.txt", mode="w")
    with ctx_in as input_sew, ctx_uout as out_sew:
        for line in tqdm(input_sew):
            line = line.strip().split()
            line_out = []
            for word in line:
                if "_bn:" in word:
                    lemmas = word.split("_")
                    syn = lemmas[-1]
                    lemma = " ".join(lemmas[:-1])
                    lemma_new = lemmatizer.lemmatize(lemma, pos=syn[-1])
                    line_out.append("_".join(lemma_new.split()) + "_" + syn)
                else:
                    line_out.append(word)
            out_sew.write(" ".join(line_out) + "\n")



def parse_all():
    """
    This method parses all used datasets whether they aren't already parsed.
    :return: None
    """
    bn2wn = build_bn2wns()

    # if Eurosense parsed sentences doesn't exists
    if not os.path.isfile(EURO_SENTENCES):
        write_file(EURO_SENTENCES, parser_eurosense(bn2wn))

    # if SEW parsed sentences doesn't exists
    if not os.path.isfile(SEW_SENTENCES):
        write_file(SEW_SENTENCES, parser_sew(bn2wn))

    # if TOM parsed sentences doesn't exists
    if not os.path.isfile(TOM_SENTENCES):
        wns2bn = build_wns2bn()
        write_file(TOM_SENTENCES, parser_tom(wns2bn))
