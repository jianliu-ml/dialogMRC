import json

from util import _find_sub_list

def change_format_all_entity_to_binary(filename, filename_out):
    with open(filename) as filein:
        data = json.loads(filein.read())
        for elem in data:
            entity_set = set()
            in_set = set()

            triples_all = list()

            for e in elem[1]:
                y, x, rid, r, t, x_type, y_type = e['y'], e['x'], e['rid'], e['r'], e['t'], e['x_type'], e['y_type']
                entity_set.add((x, x_type))
                entity_set.add((y, y_type))
                in_set.add((x, y))

                # To Do: asynsetric or synsetic relation filter

                if x[0:2] == 'Sp' or y[0:2] == 'Sp':  # at least one entity is 'Speaker'
                    triples_all.append(e)

            for c, c_type in entity_set:
                for d, d_type in entity_set:
                    if c != d and (c, d) not in in_set:  # c=x, d=y
                        if not (c[:2] == 'Sp' or d[:2] == 'Sp'):
                            continue
                        temp = {
                            'x': c,
                            'y': d,
                            'rid': [37],
                            'r': ['unanswerable'],
                            't': [''],
                            'x_type': c_type,
                            'y_type': d_type
                        }
                        triples_all.append(temp)

            elem[1] = triples_all

        fileout = open(filename_out, 'w')
        json.dump(data, fileout)
        fileout.close()

# change_format_all_entity_to_binary('data/org/train.json', 'data/train_b.json')
# change_format_all_entity_to_binary('data/org/dev.json', 'data/dev_b.json')
# change_format_all_entity_to_binary('data/org/test.json', 'data/test_b.json')


def get_all_relations(r_set):
    filename = 'data/org/train.json'
    with open(filename) as filein:
        data = json.loads(filein.read())
        for elem in data:
            for e in elem[1]:
                y, x, rid, r, t, x_type, y_type = e['y'], e['x'], e['rid'], e['r'], e['t'], e['x_type'], e['y_type']
                for r_t in r:
                    r_set.add(r_t)
# r_set = set()
# get_all_relations(r_set)
# print(sorted(list(r_set)))

# all_relations = [
#     'gpe:residents_of_place',
#     'gpe:visitors_of_place',
#     'org:employees_or_members',
#     'org:students',
#     'per:acquaintance',
#     'per:age',
#     'per:alternate_names',
#     'per:alumni',
#     'per:boss',
#     'per:children',
#     'per:client',
#     'per:date_of_birth',
#     'per:dates',
#     'per:employee_or_member_of',
#     'per:friends',
#     'per:girl/boyfriend',
#     'per:major',
#     'per:negative_impression',
#     'per:neighbor',
#     'per:origin',
#     'per:other_family',
#     'per:parents',
#     'per:pet',
#     'per:place_of_residence',
#     'per:place_of_work',
#     'per:positive_impression',
#     'per:roommate',
#     'per:schools_attended',
#     'per:siblings',
#     'per:spouse',
#     'per:subordinate',
#     'per:title',
#     'per:visited_place', 
#     'per:works', 
#     'unanswerable']


def change_format_mc(filename, filename_out):
    with open(filename) as filein:
        data = json.loads(filein.read())
        for elem in data:
            entity_set = set()
            triples_all = list()

            for e in elem[1]:
                y, x, rid, r, t, x_type, y_type = e['y'], e['x'], e['rid'], e['r'], e['t'], e['x_type'], e['y_type']
                entity_set.add((x, x_type))
                entity_set.add((y, y_type))

                if x.startswith('Speaker'):
                    for a1, a2, a3 in zip(rid, r, t):
                        if a1 != 37:
                            triples_all.append([x, y, a1, a2, a3, x_type, y_type]) ### note we put x first

            elem[1] = list(entity_set)
            elem.append(triples_all)
        
        fileout = open(filename_out, 'w')
        json.dump(data, fileout)
        fileout.close()


change_format_mc('data/org/train.json', 'data/train_mc.json')
change_format_mc('data/org/dev.json',   'data/dev_mc.json')
change_format_mc('data/org/test.json',  'data/test_mc.json')





