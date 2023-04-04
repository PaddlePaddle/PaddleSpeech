# -*- encoding:utf-8 -*-
import re
import sys
'''
@arthur: david_95

Assum you executed g2p test twice, the WER rate have some gap, you would like to see what sentences error cause your rate up.
so you may get test result ( exp/g2p )into two directories, as exp/prefolder and exp/curfolder
run this program as  "python compare_badcase.py prefolder curfolder"
then you will get diffrences between two run, uuid, phonetics, chinese samples

examples: python compare_badcase.py  exp/g2p_laotouzi  exp/g2p
in this example:  exp/g2p_laotouzi  and  exp/g2p  are two folders with two g2p tests result

'''


def compare(prefolder, curfolder):
    '''
    compare file of text.g2p.pra in two folders
    result P1 will be prefolder ; P2 will be curfolder, just about the sequence you input in argvs
    '''

    linecnt = 0
    pre_block = []
    cur_block = []
    zh_lines = []
    with open(prefolder + "/text.g2p.pra",
              "r") as pre_file, open(curfolder + "/text.g2p.pra",
                                     "r") as cur_file:
        for pre_line, cur_line in zip(pre_file, cur_file):
            linecnt += 1

            if linecnt < 11:  #skip non-data head in files
                continue
            else:
                pre_block.append(pre_line.strip())
                cur_block.append(cur_line.strip())
                if pre_line.strip().startswith(
                        "Eval:") and pre_line.strip() != cur_line.strip():
                    uuid = pre_block[-5].replace("id: (baker_",
                                                 "").replace(")", "")
                    with open("data/g2p/text", 'r') as txt:
                        conlines = txt.readlines()

                        for line in conlines:
                            if line.strip().startswith(uuid.strip()):
                                print(line)
                                zh_lines.append(re.sub(r"#[1234]", "", line))
                                break

                    print("*" + cur_block[-3])  # ref
                    print("P1 " + pre_block[-2])
                    print("P2 " + cur_block[-2])
                    print("P1 " + pre_block[-1])
                    print("P2 " + cur_block[-1] + "\n\n")
                    pre_block = []
                    cur_block = []

    print("\n")
    print(str.join("\n", zh_lines))


if __name__ == '__main__':
    assert len(
        sys.argv) == 3, "Usage: python compare_badcase.py %prefolder %curfolder"
    compare(sys.argv[1], sys.argv[2])
