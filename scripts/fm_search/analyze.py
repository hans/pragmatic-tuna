from pathlib import Path
import re
import sys

import numpy as np


sweep_path = sys.argv[1]

number_re = re.compile("[0-9]+(?:\.[0-9]+)?")
full_1_re = re.compile("L_([^\s]+): *([0-9]+(?:\.[0-9]+)?)")
full_2_re = re.compile("([A-Za-z]+[A-Za-z-]*)\s+(-?[0-9]+(?:\.[0-9]+)?)")

# Get headers
with Path(sweep_path, "0000", "stdout").open("r") as stdout0:
    for line in stdout0:
        if line.startswith("%%%%"):
            adv_headers = [header for header in line.strip(" %\t\n").split("\t")
                           if not number_re.search(header)]
            adv_headers = list(sorted(adv_headers))
        elif line.startswith("%%"):
            eval_headers = list(sorted([label for label, _ in full_1_re.findall(line)]))


header = "K\tADVFM_AVG\t"
header += "\t".join("ADVFM_%s" % s for s in adv_headers)
header += "\t" + "\t".join(eval_headers)
header += "\tID"
print(header)

results = []

for run in Path(sweep_path).glob("*"):
    run_id = run.name

    full_eval_1, full_eval_2 = None, None
    with Path(run, "stdout").open("r") as stdout_f:
        for line in stdout_f:
            if line.startswith("%%%%"):
                full_eval_2 = line
            elif line.startswith("%%"):
                full_eval_1 = line

    if full_eval_1 is None or full_eval_2 is None:
        continue

    full_1_results = dict(full_1_re.findall(full_eval_1))
    full_1_results = {k: float(x) for k, x in full_1_results.items()}
    assert set(full_1_results.keys()) == set(eval_headers)

    full_2_results = dict(full_2_re.findall(full_eval_2))
    full_2_results = {k: float(x) for k, x in full_2_results.items()}

    avg_fm = np.mean([x for k, x in full_2_results.items() if k.lower() == k])

    with Path(run, "params").open("r") as params_f:
        params = eval(params_f.read())
        k = params["fast_mapping_k"]

    results_i = [k, avg_fm]
    results_i += [full_2_results.get(el, 0) for el in adv_headers]
    results_i += [full_1_results[el] for el in eval_headers]
    results_i += [run_id]
    results.append(results_i)

print("\n".join("\t".join(map(str, results_i)) for results_i in results))
