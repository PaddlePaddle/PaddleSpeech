#!/usr/bin/python3

# from local.hyperyaml_core import load_hyperpyyaml
# from hyperpyyaml import load_hyperpyyaml
# with open("./local/test.yaml") as fin:
#     hparams = load_hyperpyyaml(fin)
# print(hparams)
# # hparams = load_hyperpyyaml("./local/test.yaml")

import speechbrain as sb

@sb.utils.data_pipeline.takes("text")
@sb.utils.data_pipeline.provides("sig")
def tokenize(text):
    print("exec the tokenize: {}".format(text))
    return text.strip().lower().split()

print(tokenize)
tokenize.provides = ["tokenized"]
print(tokenize("\tThis Example get tokenized"))