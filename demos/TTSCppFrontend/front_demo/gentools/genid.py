#from parakeet.frontend.vocab import Vocab

PHONESFILE = "./dict/phones.txt"
PHONES_ID_FILE = "./dict/phonesid.dict"
TONESFILE = "./dict/tones.txt"
TONES_ID_FILE = "./dict/tonesid.dict"

def GenIdFile(file, idfile):
    id = 2
    with open(file, 'r') as f1, open(idfile, "w+") as f2:
        f2.write("<pad> 0\n")
        f2.write("<unk> 1\n")
        for line in f1.readlines():
            phone = line.strip()
            print(phone + " " + str(id) + "\n")
            f2.write(phone + " " + str(id) + "\n")
            id += 1

if __name__ == "__main__":
    GenIdFile(PHONESFILE, PHONES_ID_FILE)
    GenIdFile(TONESFILE, TONES_ID_FILE)

