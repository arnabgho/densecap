import json
import operator
import argparse

def main(opt):

    with open(opt.qas,'r') as f:
        qas_json=json.load(f)

    subject_dict={}
    object_dict={}
    predicate_dict={}
    answer_dict={}

    for rel in qas_json:
        for rels in rel['qas']:
          #  predicate_dict[ rels['predicate']]=predicate_dict.get(rels['predicate'],0)+1
          #  subject_dict[ rels['subject']['name']]=subject_dict.get(rels['subject']['name'] ,0)+1
          #  object_dict[ rels['object'][ 'name'] ]=object_dict.get(rels['object']['name'],0)+1
            answer_dict[ rels[ 'answer'  ]   ] = answer_dict.get( rels['answer']  , 0 )+1


   # sorted_predicates=sorted(predicate_dict.items()  , key=operator.itemgetter(1) , reverse=True)
   # sorted_subjects=sorted(subject_dict.items()  , key=operator.itemgetter(1) , reverse=True)
   # sorted_objects=sorted(object_dict.items()  , key=operator.itemgetter(1) , reverse=True)
    sorted_stats=sorted(answer_dict.items()  , key=operator.itemgetter(1) , reverse=True)




    for rel in sorted_stats:
        print rel

if __name__== '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--qas',
                default='../data/jsons/question_answers.json',
                help="The qas JSON File")

    opt=parser.parse_args()
    main(opt)

