import sys
import json
from argparse import ArgumentParser


def main(args):
    f = open(args.corpus_path, 'r')
    
    data = json.load(f)
    
    num_duplicates = 0
    
    for trial in data:
        domain = trial["domain"]
        domain_sorted = sorted(domain, key=lambda x: x['target'], reverse=True)
        entries = set()
        new_domain = []
        for subgraph in domain_sorted:
            sig = "%s-%s-%s" % (subgraph['object1'], subgraph['reln'], subgraph['object2'])
            if sig not in entries:
                new_domain.append(subgraph)
                entries.add(sig)
            else:
                num_duplicates += 1
    
        trial["domain"] = new_domain
    
    json.dump(data, sys.stdout, f, indent=4)
        
    print("Number of duplicates: %d" % num_duplicates, file=sys.stderr)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("corpus_path")

    main(p.parse_args())
