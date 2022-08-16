#!/usr/bin/env python3
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__doc__)
    parser.add_argument(
        '--logfile', type=str, required=True, help='ws client log file')

    args = parser.parse_args()

    rtfs = []
    with open(args.logfile, 'r') as f:
        for line in f:
            if 'RTF=' in line:
                # udio duration: 6.126, elapsed time: 3.471978187561035, RTF=0.5667610492264177
                line = line.strip()
                beg = line.index("audio")
                line = line[beg:]

                items = line.split(',')
                vals = []
                for elem in items:
                    if "RTF=" in elem:
                        continue
                    _, val = elem.split(":")
                    vals.append(eval(val))
                keys = ['T', 'P']
                meta = dict(zip(keys, vals))

                rtfs.append(meta)

    T = 0.0
    P = 0.0
    n = 0
    for m in rtfs:
        # not accurate, may have duplicate log
        n += 1  
        T += m['T']
        P += m['P']

    print(f"RTF: {P/T}, utts: {n}")
