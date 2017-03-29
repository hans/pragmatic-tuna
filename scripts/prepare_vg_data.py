#!/usr/bin/python


from argparse import ArgumentParser
from jsonstreamer import ObjectStreamer
import sys


TRAIN_RELATIONS = ["along.r.01", "in.r.01"]
FAST_MAPPING_RELATIONS = ["behind.r.01"]


class VisualGenomeFilter(object):
    
    def __init__(self):
        self.train_candidates = set()
        self.fast_mapping_candidates = set()
        self.train_set = set()
        self.fast_mapping_set = set()
        self.known_objects = set()
        self.fast_mapping_trials = set()
        self.trials = set()

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
                    synsets = reln['synsets']
                    for synset in synsets:
                        if synset in TRAIN_RELATIONS:
                            self.train_candidates.add(image_id)
                            break
                    for synset in synsets:
                        if synset in FAST_MAPPING_RELATIONS:
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
                        synsets = reln['synsets']
                        for synset in synsets:
                            if synset in TRAIN_RELATIONS:
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
                        synsets = reln['synsets']
                        for synset in synsets:
                            if synset in TRAIN_RELATIONS or synset in FAST_MAPPING_RELATIONS:
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
                    self.fast_mapping_trials.add(image_id)


    def _convert_reln_to_domain_entry(self, reln, objects, is_target=False):
        entry = {}
        entry['object1'] = objects[reln['subject_id']]
        entry['object2'] = objects[reln['object_id']]
        entry['reln'] = reln['synsets'][0]
        entry['target'] = is_target
        return entry

    def _get_objects(self, region):
        return {obj['object_id']: obj['synsets'][0] 
                        for obj in region['objects'] if len(obj['synsets']) > 0}

    def _trial_listener(self, event, *args):
        if event == "element":
            image = args[0]
            image_id = int(image['image_id'])
            has_target = False
            utterance = None
            domain = []

            #add training trial
            if image_id in train_set:
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue 
                    objects = self._get_objects(region)
                    for reln in region['relationships']:
                        synsets = reln['synsets']
                        if len(synsets) > 0 and synsets[0] in TRAIN_RELATIONS:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects, not has_target))
                            if not has_target:
                                has_target = True
                                utterance = region['phrase']
                        elif len(synsets) > 0:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects))
                if has_target:
                    trial = {}
                    trial['type'] = "train"
                    trial['utterance'] = utterance
                    trial['domain'] = domain
                    self.trials.append(trial)

            # add fast mapping trial
            elif image_id in self.fast_mapping_trials:
                for region in image['regions']:
                    if len(region['relationships']) != 1:
                        continue 
                    objects = self._get_objects(region)
                    for reln in region['relationships']:
                        synsets = reln['synsets']
                        if len(synsets) > 0 and synsets[0] in FAST_MAPPING_RELATIONS:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects, not has_target))
                            if not has_target:
                                has_target = True
                                utterance = region['phrase']
                        elif len(synsets) > 0 and synsets[0] in TRAIN_RELATIONS:
                            if reln['object_id'] not in objects or reln['subject_id'] not in objects:
                                continue
                            domain.append(self._convert_reln_to_domain_entry(reln, objects))
                if has_target:
                    trial = {}
                    trial['type'] = "fast_mapping"
                    trial['utterance'] = utterance
                    trial['domain'] = domain
                    self.trials.append(trial)

    def main(self, args):

        f = open(args.corpus_path, 'r')

        object_streamer = ObjectStreamer()
        object_streamer.add_catch_all_listener(self._filter_candidates)
        data = f.read(100000)
        while data != "":
            object_streamer.consume(data)
            data = f.read(100000)

        self.train_set = self.train_candidates.difference(self.fast_mapping_candidates)
        self.fast_mapping_set = self.fast_mapping_candidates.intersection(self.train_candidates)

        f.seek(0)
        object_streamer = ObjectStreamer()
        object_streamer.add_catch_all_listener(self._store_known_objects)
        data = f.read(100000)
        while data != "":
            object_streamer.consume(data)
            data = f.read(100000)

        f.seek(0)
        object_streamer = ObjectStreamer()
        object_streamer.add_catch_all_listener(self._filter_fast_mapping_trials)
        data = f.read(100000)
        while data != "":
            object_streamer.consume(data)
            data = f.read(100000)


        f.seek(0)
        object_streamer = ObjectStreamer()
        object_streamer.add_catch_all_listener(self._trial_listener)
        data = f.read(100000)
        while data != "":
            object_streamer.consume(data)
            data = f.read(100000)


        json.dump(self.trials, sys.stdout, indent=4)



if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("corpus_path")

    filt = VisualGenomeFilter()

    filt.main(p.parse_args())
