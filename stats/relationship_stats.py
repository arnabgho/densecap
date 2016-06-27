import json
import operator
import argparse

def main(opt):

    with open(opt.relationship,'r') as f:
        relationship_json=json.load(f)

    subject_dict={}
    object_dict={}
    predicate_dict={}

    for rel in relationship_json:
        for rels in rel['relationships']:
          #  predicate_dict[ rels['predicate']]=predicate_dict.get(rels['predicate'],0)+1
          #  subject_dict[ rels['subject']['name']]=subject_dict.get(rels['subject']['name'] ,0)+1
            object_dict[ rels['object'][ 'name'] ]=object_dict.get(rels['object']['name'],0)+1



   # sorted_predicates=sorted(predicate_dict.items()  , key=operator.itemgetter(1) , reverse=True)
   # sorted_subjects=sorted(subject_dict.items()  , key=operator.itemgetter(1) , reverse=True)
    sorted_objects=sorted(object_dict.items()  , key=operator.itemgetter(1) , reverse=True)



    for rel in sorted_objects:
        print rel

if __name__== '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--relationship',
                default='../data/jsons/relationships.json',
                help="The Relationship JSON File")

    opt=parser.parse_args()
    main(opt)

