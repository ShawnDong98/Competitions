import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CNTokenizer:
    def __init__(
        self,
        max_length,
    ):
        self.max_length = max_length

    def build_vocab(
        self,
        captions,
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    ):
        s = []
        for caption in tqdm(captions):
            s.extend(list(caption))
            # s.extend([token.text for token in self.spacy_en(caption)])
        words = sorted(list(set(s)))
        self.stoi = { ch: i+4 for i, ch in enumerate(words) }
        self.itos = { i+4: ch for i, ch in enumerate(words) }

        for i, ch in enumerate(special_tokens):
            self.stoi[ch] = i
            self.itos[i] = ch
        

    def __call__(
        self, 
        sentence, 
        padding=True, 
        truncation=True, 
        return_tokens=True
    ):
        list_sentence = list(sentence)
        logger.debug(list_sentence)
        list_ids = [self.stoi.get(ch, self.stoi['[UNK]']) for ch in list_sentence]
        logger.debug(list_ids)

        if truncation:
            if len(list_ids) > self.max_length-2:
                list_ids = list_ids[:self.max_length-2]

        list_ids = self.PostProcess(list_ids)

        if padding:
            if len(list_ids) < self.max_length:
                list_ids.extend([self.stoi['[PAD]']] * (self.max_length - len(list_ids)))
        
        
        if return_tokens:
            list_tokens = self.decode(list_ids)
            return list_ids, list_tokens

        return list_ids

    def PostProcess(self, list_ids):
        """
        在tokenizer的句子前面加[CLS]， 后面加[SEP]
        args: 
            code : 
         
        return: 
            code : 
        """
        post_list_ids = [self.stoi['[CLS]']]
        post_list_ids.extend(list_ids)
        post_list_ids.append(self.stoi['[SEP]'])

        return post_list_ids

    def decode(self, ids):
        return [self.itos[id_] for id_ in ids]

    def load_vocab(self, vocab_file):
        import json
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)

        self.itos = { i: ch for ch, i in self.stoi.items()}

    def save_vocab(self, vocab_file):
        import json
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.stoi, f, ensure_ascii=False)

    def vocab_size(self):
        return len(self.stoi)



class ENTokenizer:
    def __init__(
        self,
        max_length,
    ):
        import spacy
        self.max_length = max_length
        self.spacy_en = spacy.load("en_core_web_sm")

    def build_vocab(
        self,
        captions,
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    ):
        s = []
        for caption in tqdm(captions):
            s.extend(list(self.spacy_en(caption)))
            # s.extend([token.text for token in self.spacy_en(caption)])
        words = sorted(list(set(s)))
        self.stoi = { ch: i+4 for i, ch in enumerate(words) }
        self.itos = { i+4: ch for i, ch in enumerate(words) }

        for i, ch in enumerate(special_tokens):
            self.stoi[ch] = i
            self.itos[i] = ch

    def __call__(
        self,
        sentence,
        padding = True,
        truncation = True,
        return_tokens = False
    ):
        tokens = [token.text for token in self.spacy_en(sentence)]
        ids = [self.stoi.get(token, self.stoi.get("[UNK]")) for token in tokens]

        if truncation == True:
            if len(ids) > self.max_length - 2:
                ids = ids[:self.max_length - 2]

        post_ids = self.postprocess(ids)

        if padding == True:
            if len(post_ids) < self.max_length:
                post_ids.extend([self.stoi['[PAD]']] * (self.max_length - len(post_ids)))

        if return_tokens == True:
            tokens = " ".join(self.decode(post_ids))
            return post_ids, tokens

        return post_ids


    def postprocess(
        self,
        ids
    ):
        post_ids = [self.stoi["[CLS]"]]
        post_ids.extend(ids)
        post_ids.append(self.stoi["[SEP]"])

        return post_ids


    def decode(
        self,
        post_ids
    ):
        return [self.itos[id_] for id_ in post_ids]

    def save_vocab(
        self,
        vocab_file
    ):
        import json
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False)


    def load_vocab(
        self,
        vocab_file
    ):
        import json
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)

        self.itos = { i: ch for ch, i in self.stoi.items()}


    def vocab_size(self):
        return len(self.stoi)