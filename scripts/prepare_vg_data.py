#!/usr/bin/python


from argparse import ArgumentParser
from jsonstreamer import ObjectStreamer
import sys
import json
import copy

TRAIN_RELATIONS = ["on", "in"]
FAST_MAPPING_RELATIONS = ["behind"]

DEV_SPLIT_SIZE = 0.1
TEST_SPLIT_SIZE = 0.1


class VisualGenomeFilter(object):

    def __init__(self):
        self.train_candidates = set()
        self.fast_mapping_candidates = set()
        self.train_set = set()
        self.dev_set = set()
        self.test_set = set()
        self.fast_mapping_set = set()
        self.known_objects = set()
        self.fast_mapping_train_set = set()
        self.trials = []
        self._cache = None

    # adds all images that contain a region with at least one of the relations in TRAIN_RELATION
    # to train_candidates, and images w/ a region with a relation in FAST_MAPPING_RELATIONS
    # to fast_mapping_candidates.
    def _filter_candidates(self, event, *args):
        if event == "element":
            image = args[0]
            image_id = int(image['image_id'])
            for region in image['regions']:
                if len(region['relationships']) != 1:
                    continue
                for reln in region['relationships']:
                    predicate = reln['predicate'].lower()
                    if predicate in TRAIN_RELATIONS:
                        self.train_candidates.add(image_id)
                        break
                    if predicate in FAST_MAPPING_RELATIONS:
                        self.fast_mapping_candidates.add(image_id)
                        break


    # stores synsets of object that appear in pre-training data
    def _store_known_objects(self, event, *args):
        if event == "element":
            image = args[0]
            image_id = int(image['image_id'])
            if image_id in self.train_set:
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue
                    for reln in region['relationships']:
                        predicate = reln['predicate'].lower()
                        if predicate in TRAIN_RELATIONS:
                            objects = {obj['object_id']: obj['synsets'][0] 
                                            for obj in region['objects'] if len(obj['synsets']) > 0}
                            
                            if reln['subject_id'] in objects and reln['object_id'] in objects:
                                self.known_objects.add(objects[reln['subject_id']])
                                self.known_objects.add(objects[reln['object_id']])

    # store all the fast mapping trials
    def _filter_fast_mapping_trials(self, event, *args):
        if event == "element":
            image = args[0]
            image_id = int(image['image_id'])
            if image_id in self.fast_mapping_set:
                include = True
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue
                    for reln in region['relationships']:
                        predicate = reln['predicate'].lower()
                        if predicate in TRAIN_RELATIONS or predicate in FAST_MAPPING_RELATIONS:
                            objects = self._get_objects(region)
                            if reln['subject_id'] not in objects or objects[reln['subject_id']] not in self.known_objects:
                                include = False
                                break
                            elif reln['object_id'] not in objects or objects[reln['object_id']] not in self.known_objects:
                                include = False
                                break
                    if not include:
                        break
                if include:
                    self.fast_mapping_train_set.add(image_id)


    def _filter_pre_train_trials(self, event, *args):
        if event == "element":
            image = args[0]
            image_id = int(image['image_id'])
            is_dev = image_id in self.dev_candidates
            is_test = image_id in self.test_candidates
            if is_dev or is_test:
                include = True
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue
                    for reln in region['relationships']:
                        predicate = reln['predicate'].lower()
                        if predicate in TRAIN_RELATIONS:
                            objects = self._get_objects(region)
                            if reln['subject_id'] not in objects or objects[reln['subject_id']] not in self.known_objects:
                                include = False
                                break
                            elif reln['object_id'] not in objects or objects[reln['object_id']] not in self.known_objects:
                                include = False
                                break
                    if not include:
                        break
                if include:
                    if is_dev:
                        self.dev_set.add(image_id)
                    else:
                        self.test_set.add(image_id)

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

    def _create_adverserial_trial(self, trial):
        for entry in trial['domain']:
            if entry['target']:
                target = entry
                break
        
        adv_trial = copy.deepcopy(trial)
        adv_trial['type'] = 'adv_fast_mapping'
        domain = []
        domain.append(target)
        for reln in TRAIN_RELATIONS:
            entry = copy.deepcopy(target)
            entry['reln'] = reln
            entry['target'] = False
            domain.append(entry)
        adv_trial['domain'] = domain
        return adv_trial

    def _trial_listener(self, event, *args):
        if event == "element":
            image = args[0]
            image_id = int(image['image_id'])
            has_target = False
            utterance = None
            domain = []
            
            t = None
            if image_id in self.train_set:
                pre_train = True
                t = "train"
            elif image_id in self.dev_set:
                pre_train = True
                t = "dev"
            elif image_id in self.test_set:
                pre_train = True
                t = "test"
            elif image_id in self.fast_mapping_train_set:
                pre_train = False
                t = "train"
            elif image_id in self.fast_mapping_dev_set:
                pre_train = False
                t = "dev"
            elif image_id in self.fast_mapping_test_set:
                pre_train = False
                t = "test"
            
            #add training trial
            if t is not None and pre_train:
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue
                    objects = self._get_objects(region)
                    for reln in region['relationships']:
                        predicate = reln['predicate'].lower()
                        if predicate in TRAIN_RELATIONS:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects, not has_target))
                            if not has_target:
                                has_target = True
                                utterance = region['phrase']
                        else:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects))
                if has_target:
                    trial = {}
                    trial['type'] = "pre_train_%s" % t
                    trial['utterance'] = utterance.lower()
                    trial['domain'] = domain
                    self.trials.append(trial)

            # add fast mapping trial
            elif t is not None:
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue
                    objects = self._get_objects(region)
                    for reln in region['relationships']:
                        predicate = reln['predicate'].lower()
                        if predicate in FAST_MAPPING_RELATIONS:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects, not has_target))
                            if not has_target:
                                has_target = True
                                utterance = region['phrase']
                        elif t != "train" or predicate in TRAIN_RELATIONS:
                            # NB: extra constraint in the fast mapping trials:
                            # only include relations observed during training,
                            # or the fast_mapping_ast-mapping relation
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects))
                if has_target:
                    trial = {}
                    trial['type'] = "fast_mapping_%s" % t
                    trial['utterance'] = utterance.lower()
                    trial['domain'] = domain
                    self.trials.append(trial)
                    if t == "train":
                        adv_trial = self._create_adverserial_trial(trial)
                        self.trials.append(adv_trial)

    def _apply_object_stream_function_to_json_file(self, f, function, load_in_memory=False):
        
        if load_in_memory:
            if self._cache is None:
                self._cache = json.load(f)
            for element in self._cache:
                function("element", [element])
        else:
            f.seek(0)
            object_streamer = ObjectStreamer()
            object_streamer.add_catch_all_listener(function)
            data = f.read(100000)
            i = 0
            while data != "":
                i += 1
                print(i, file=sys.stderr)
                object_streamer.consume(data)
                data = f.read(100000)

    def main(self, args):

        f = open(args.corpus_path, 'r')

        load_in_memory = args.load_in_memory

        self._apply_object_stream_function_to_json_file(f, self._filter_candidates, load_in_memory)

        self.train_set = self.train_candidates.difference(self.fast_mapping_candidates)

        self.dev_candidates = set()
        self.test_candidates = set()

        train_size = len(self.train_set)

        # train/dev/test split for pre-training
        for i, img in enumerate(self.train_set):
            if i < train_size * DEV_SPLIT_SIZE:
                self.dev_candidates.add(img)
            elif i < train_size * (DEV_SPLIT_SIZE + TEST_SPLIT_SIZE):
                self.test_candidates.add(img)
        
        self.train_set = self.train_set.difference(self.dev_candidates).difference(self.test_candidates)

        self.fast_mapping_set = self.fast_mapping_candidates.intersection(self.train_candidates)

        self._apply_object_stream_function_to_json_file(f, self._store_known_objects, load_in_memory)

        # filter dev/test set for pre-training so that all target objects are known
        self._apply_object_stream_function_to_json_file(f, self._filter_pre_train_trials, load_in_memory)

        self._apply_object_stream_function_to_json_file(f, self._filter_fast_mapping_trials, load_in_memory)

        # train/dev/test split for fast mapping
        fast_mapping_size = len(self.fast_mapping_train_set)

        self.fast_mapping_dev_set = set()
        self.fast_mapping_test_set = set()

        for i, img in enumerate(self.fast_mapping_train_set):
            if i < fast_mapping_size * DEV_SPLIT_SIZE:
                self.fast_mapping_dev_set.add(img)
            elif i < fast_mapping_size * (DEV_SPLIT_SIZE + TEST_SPLIT_SIZE):
                self.fast_mapping_test_set.add(img)

        self.fast_mapping_train_set = self.fast_mapping_train_set.\
                                               difference(self.fast_mapping_dev_set).\
                                               difference(self.fast_mapping_test_set)

        self._apply_object_stream_function_to_json_file(f, self._trial_listener, load_in_memory)

        json.dump(self.trials, sys.stdout, indent=4)



if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("corpus_path")
    p.add_argument("--load_in_memory", action="store_true", default=False)

    filt = VisualGenomeFilter()

    filt.main(p.parse_args())
