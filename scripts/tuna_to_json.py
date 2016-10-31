# Convert the TUNA XML corpus to a friendlier JSON format.

from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import os.path
import sys
from xml.etree import ElementTree as ET

import nltk


def parse_domain_entity(entity_el):
    entity = {
        "id": entity_el.get("ID"),
        "image": entity_el.get("IMAGE"),
        "target": entity_el.get("TYPE") == "target",
        "attributes": {attr_el.get("NAME"): get_attribute_value(attr_el)
                       for attr_el in entity_el.findall("./ATTRIBUTE")},
    }

    return entity


def get_attribute_value(attribute_el):
    attr_type = attribute_el.get("TYPE")
    attr_value = attribute_el.get("VALUE")
    if attr_type == "literal":
        return attr_value
    elif attr_type == "gradable":
        try:
            return int(attr_value)
        except ValueError:
            return None
    elif attr_type == "boolean":
        return bool(int(attr_value))
    else:
        raise ValueError("unknown attribute type %s" % attr_type)


def parse_trial(path):
    tree = ET.parse(path).getroot()

    domain = [parse_domain_entity(entity_el)
              for entity_el in tree.findall("./DOMAIN/ENTITY")]

    return {
        "id": tree.get("ID"),
        "cardinality": int(tree.get("CARDINALITY")),
        "condition": tree.get("CONDITION"),
        "similarity": tree.get("SIMILARITY"),

        "domain": domain,
        "string_description": tree.findtext("./STRING-DESCRIPTION").strip().lower(),
    }


def main(args):
    corpora = {}
    for corpus in "furniture", "people":
        path = os.path.join(args.corpus_path, "corpus", "singular", corpus)
        trials = [parse_trial(os.path.join(path, trial_path))
                  for trial_path in os.listdir(path) if trial_path.endswith(".xml")]

        # Compute trial and attribute metadata.
        domain_size = len(trials[0]["domain"])
        attribute_values = defaultdict(set)
        vocab = set()
        for trial in trials:
            assert len(trial["domain"]) == domain_size

            for item in trial["domain"]:
                for attribute, value in item["attributes"].items():
                    attribute_values[attribute].add(value)

            tokens = nltk.word_tokenize(trial["string_description"])
            vocab |= set(tokens)

        corpora[corpus] = {
            "attributes": {key: list(values) for key, values
                           in attribute_values.items()},
            "vocab": sorted(vocab),
            "trials": trials,
        }

    json.dump(corpora, sys.stdout, sort_keys=True, indent=4)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("corpus_path")

    main(p.parse_args())
