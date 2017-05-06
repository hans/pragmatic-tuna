#!/usr/bin/python


from argparse import ArgumentParser
from collections import defaultdict, Counter
import sys
import json
import copy
import random

try:
    from jsonstreamer import ObjectStreamer
except:
    pass


POSSIBLE_TRAIN_RELATIONS = ["on", "in"]
POSSIBLE_FAST_MAPPING_RELATIONS = ["behind", "near", "under"]

DEV_SPLIT_SIZE = 0.1
TEST_SPLIT_SIZE = 0.1

FM_DEV_SPLIT_SIZE = 0.33
FM_TEST_SPLIT_SIZE = 0.33


class VisualGenomeFilter(object):

    def __init__(self):
        self.TRAIN_RELATIONS = []
        self.FM_RELATIONS = []
        self._cache = None
        self._relationindex = defaultdict(set)
        self.splits = defaultdict(set)
        self.known_objects = set()
        self.trials = []
        self.corpora = defaultdict(list)

    # populates self.TRAIN_RELATIONS and self.FM_RELATIONS with most frequent
    # relations
    def _populate_relations(self, train_relation_count, fm_relation_count):
        self.TRAIN_RELATIONS = POSSIBLE_TRAIN_RELATIONS[0:train_relation_count]
        self.FM_RELATIONS = POSSIBLE_FAST_MAPPING_RELATIONS[0:fm_relation_count]


    def _index_relations(self):
        for image_id, image in self._cache.items():
            for region in image['regions']:
                if len(region['relationships']) != 1:
                    continue
                reln = region['relationships'][0]
                predicate = reln['predicate'].lower()

                if predicate in self.TRAIN_RELATIONS \
                        or predicate in self.FM_RELATIONS:
                    self._relationindex[predicate].add(image_id)

    def _construct_splits(self):
        train_candidates = set()
        for reln in self.TRAIN_RELATIONS:
            train_candidates = train_candidates.union(self._relationindex[reln])

        fm_candidates = set()
        for reln in self.FM_RELATIONS:
            fm_candidates = fm_candidates.union(self._relationindex[reln])

        pt_all = train_candidates.difference(fm_candidates)
        pt_all_size = len(pt_all)

        fm_all = fm_candidates.intersection(train_candidates)
        fm_all_size = len(fm_all)

        for i, img_id in enumerate(pt_all):
            if i < pt_all_size * DEV_SPLIT_SIZE:
                self.splits["pre_train_dev"].add(img_id)
            elif i < pt_all_size * (DEV_SPLIT_SIZE + TEST_SPLIT_SIZE):
                self.splits["pre_train_test"].add(img_id)
            else:
                self.splits["pre_train_train"].add(img_id)

        for i, img_id in enumerate(fm_all):
            if i < fm_all_size * FM_DEV_SPLIT_SIZE:
                self.splits["fast_mapping_dev"].add(img_id)
            elif i < fm_all_size * (FM_DEV_SPLIT_SIZE + FM_TEST_SPLIT_SIZE):
                self.splits["fast_mapping_test"].add(img_id)
            else:
                self.splits["fast_mapping_train"].add(img_id)

        print("Size of splits:", file=sys.stderr)

        for x in self.splits:
          print(x, file=sys.stderr)
          print(len(self.splits[x]), file=sys.stderr)


    # make sure that none of the relations in FM_RELATIONS appear within a
    # relation or an utterance of the pre_train_train split
    def _filter_pt_train(self):
        new_pt_train = set()
        for image_id in self.splits["pre_train_train"]:
            image = self._cache[image_id]
            include = True
            for region in image['regions']:
                if len(region['relationships']) != 1:
                    continue
                reln = region['relationships'][0]
                predicate = reln['predicate'].lower()
                predicate_tokens = predicate.split()
                for fm_reln in self.FM_RELATIONS:
                    if fm_reln in predicate_tokens:
                        include = False
                        break

                if include and predicate in self.TRAIN_RELATIONS:
                    utterance_tokens = region["phrase"].lower().split()
                    for fm_reln in self.FM_RELATIONS:
                        if fm_reln in utterance_tokens:
                            include = False
                            break
                if not include:
                    break

            if include:
                new_pt_train.add(image_id)

        self.splits["pre_train_train"] = new_pt_train


    def _store_known_objects(self):
        for image_id in self.splits["pre_train_train"]:
            image = self._cache[image_id]
            for region in image['regions']:
                if len(region['relationships']) != 1:
                    continue
                reln = region['relationships'][0]
                predicate = reln['predicate'].lower()
                if predicate in self.TRAIN_RELATIONS:
                    objects = self._get_objects(region)
                    if reln['subject_id'] in objects \
                            and reln['object_id'] in objects:
                        self.known_objects.add(objects[reln['subject_id']])
                        self.known_objects.add(objects[reln['object_id']])


    # filters pre-train dev/test trials so that the target only contains
    # previously observed objects
    def _filter_pre_train(self):
        for split in ["pre_train_dev", "pre_train_test"]:
            new_split = set()
            for image_id in self.splits[split]:
                image = self._cache[image_id]
                include = True
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue
                    reln = region['relationships'][0]
                    predicate = reln['predicate'].lower()
                    if predicate in self.TRAIN_RELATIONS:
                        objects = self._get_objects(region)
                        if reln['subject_id'] not in objects \
                                or objects[reln['subject_id']] not in self.known_objects:
                            include = False
                            break
                        elif reln['object_id'] not in objects \
                                or objects[reln['object_id']] not in self.known_objects:
                            include = False
                            break

                if include:
                    new_split.add(image_id)

            self.splits[split] = new_split


    # filters fast mapping train/dev/test trials so that the target and
    # distractors only contain previously observed objects
    def _filter_fm(self):
        for split in ["fast_mapping_train", "fast_mapping_dev", "fast_mapping_test"]:
            new_split = set()
            for image_id in self.splits[split]:
                image = self._cache[image_id]
                include = True
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue
                    reln = region['relationships'][0]
                    predicate = reln['predicate'].lower()
                    if predicate in self.TRAIN_RELATIONS or predicate in self.FM_RELATIONS:
                        objects = self._get_objects(region)
                        if reln['subject_id'] not in objects \
                                or objects[reln['subject_id']] not in self.known_objects:
                            include = False
                            break
                        elif reln['object_id'] not in objects \
                                or objects[reln['object_id']] not in self.known_objects:
                            include = False
                            break

                if include:
                    new_split.add(image_id)

            self.splits[split] = new_split



    def _convert_reln_to_domain_entry(self, reln, objects, is_target=False):
        entry = {}
        entry['object1'] = objects[reln['subject_id']]
        entry['object2'] = objects[reln['object_id']]
        entry['reln'] = reln['predicate'].lower()
        entry['target'] = is_target
        return entry

    def _get_objects(self, region):
        return {obj['object_id']: obj['synsets'][0]
                        for obj in region['objects'] if len(obj['synsets']) > 0}

    def _create_adverserial_trial(self, trial, t, other_relations=self.TRAIN_RELATIONS):
        for entry in trial['domain']:
            if entry['target']:
                target = entry
                break

        adv_trial = copy.deepcopy(trial)
        adv_trial['type'] = 'adv_fast_mapping_%s' % t
        domain = []
        domain.append(target)
        for reln in other_relations:
            entry = copy.deepcopy(target)
            entry['reln'] = reln
            entry['target'] = False
            domain.append(entry)

        new_domain = list(domain)
        #for entry in domain:
        #    entry_rev = copy.deepcopy(entry)
        #    entry_rev['object1'] = entry['object2']
        #    entry_rev['object2'] = entry['object1']
        #    entry_rev['target'] = False
        #    new_domain.append(entry_rev)

        adv_trial['domain'] = new_domain
        return adv_trial


    def _generate_trials(self):
        for t in ["pre_train", "fast_mapping"]:

            target_relations = self.TRAIN_RELATIONS if t == "pre_train" else self.FM_RELATIONS

            for split in ["train", "dev", "test"]:
                split_name = t + "_" + split
                for image_id in self.splits[split_name]:
                    image = self._cache[image_id]
                    has_target = False
                    utterance = None
                    target_reln = None
                    domain = []
                    for region in image['regions']:
                        if len(region['relationships']) != 1:
                            continue
                        objects = self._get_objects(region)
                        reln = region['relationships'][0]
                        predicate = reln['predicate'].lower()
                        if predicate in target_relations:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects, not has_target))
                            if not has_target:
                                has_target = True
                                utterance = region['phrase']
                                target_reln = predicate

                        else:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            if split_name == "fast_mapping_train" and predicate not in self.TRAIN_RELATIONS:
                                continue

                            domain.append(self._convert_reln_to_domain_entry(reln, objects))

                    if has_target:

                        if t == "fast_mapping":
                            trial = {}
                            trial['type'] = "dreaming_%s" % split
                            trial['utterance'] = utterance.lower()
                            trial['domain'] = domain
                            self.trials.append(trial)

                            fm_trial = copy.deepcopy(trial)
                            fm_trial['type'] = split_name
                            fm_trial['domain'] = [x for x in domain if x['reln'] in self.TRAIN_RELATIONS or x['reln'] in self.FM_RELATIONS]
                            self.trials.append(fm_trial)

                            other_relations = set(self.FAST_MAPPING_RELATIONS + self.TRAIN_RELATIONS).difference(set([target_reln]))
                            adv_trial = self._create_adverserial_trial(trial, split)
                            self.trials.append(adv_trial)
                            self.corpora[adv_trial['type']].append(adv_trial)
                        else:
                            trial = {}
                            trial['type'] = split_name
                            trial['utterance'] = utterance.lower()
                            trial['domain'] = domain
                            self.trials.append(trial)
                            self.corpora[split_name + "_" + target_reln].append(trial)



    def _add_pre_train_adv_trials(self):
        for split in ["train", "dev", "test"]:
            adv_corpus_name = "adv_fast_mapping_%s" % split
            corpus_len = len(self.corpora[adv_corpus_name])
            for reln in self.TRAIN_RELATIONS:
                train_trials_corpus_name = "pre_train_%s_%s" % (split, reln)
                train_trials = self.corpora[train_trials_corpus_name]
                k = min(corpus_len, len(train_trials))
                adv_trials = random.sample(train_trials, k)
                other_relations = set(self.FAST_MAPPING_RELATIONS + self.TRAIN_RELATIONS).difference(set([reln]))
                for trial in adv_trials:
                    adv_trial = self._create_adverserial_trial(trial, split,other_relations=other_relations)
                    self.trials.append(adv_trial)
                    self.corpora[adv_trial['type']].append(adv_trial)


    def main(self, args):

        f = open(args.corpus_path, 'r')

        data = json.load(f)

        self._cache = dict()
        for image in data:
            self._cache[image["image_id"]] = image

        self._populate_relations(args.train_relations, args.fast_mapping_relations)
        self._index_relations()
        self._construct_splits()
        self._filter_pt_train()
        self._store_known_objects()
        self._filter_pre_train()
        self._filter_fm()
        self._generate_trials()
        self._add_pre_train_adv_trials()

        json.dump(self.trials, sys.stdout, indent=4)



if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("corpus_path")

    p.add_argument("--train_relations", type=int, default=2)
    p.add_argument("--fast_mapping_relations", type=int, default=1)


    filt = VisualGenomeFilter()

    filt.main(p.parse_args())
