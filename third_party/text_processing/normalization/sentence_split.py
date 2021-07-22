import re
from typing import List


SENTENCE_SPLITOR = re.compile(r'([。！？][”’]?)')

def split(text: str) -> List[str]:
    """Split long text into sentences with sentence-splitting punctuations.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    List[str]
        Sentences.
    """
    text = SENTENCE_SPLITOR.sub(r'\1\n', text)
    text = text.strip()
    sentences = [sentence.strip() for sentence in re.split(r'\n+', text)]
    return sentences
