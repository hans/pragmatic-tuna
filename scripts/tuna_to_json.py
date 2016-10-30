# Convert the TUNA XML corpus to a friendlier JSON format.

from argparse import ArgumentParser
import json
import os
import os.path
import sys
from xml.etree import ElementTree as ET


def parse_domain_entity(entity_el):
    entity = {
        "id": entity_el.get("ID"),
        "image": entity_el.get("IMAGE"),
        "type": entity_el.get("TYPE"),
    }

    attributes = {attr_el.get("NAME"): get_attribute_value(attr_el)
                    for attr_el in entity_el.findall("./ATTRIBUTE")}
    entity.update(attributes)

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
        "cardinality": tree.get("CARDINALITY"),
        "condition": tree.get("CONDITION"),
        "similarity": tree.get("SIMILARITY"),

        "domain": domain,
        "string_description": tree.findtext("./STRING-DESCRIPTION").strip(),
    }


def main(args):
    corpora = {}
    for corpus in "furniture", "people":
        path = os.path.join(args.corpus_path, "corpus", "singular", corpus)
        trials = [parse_trial(os.path.join(path, trial_path))
                  for trial_path in os.listdir(path) if trial_path.endswith(".xml")]

        corpora[corpus] = trials

    json.dump(corpora, sys.stdout, sort_keys=True, indent=4)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("corpus_path")

    main(p.parse_args())
