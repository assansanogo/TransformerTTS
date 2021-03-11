from typing import Union

from phonemizer.phonemize import phonemize

from preprocessing.text.symbols import all_phonemes


class Tokenizer:
    
    def __init__(self, start_token='>', 
                 end_token='<', 
                 pad_token='/', 
                 add_start_end=True, 
                 alphabet=None):
        
        # alphabet = all phonemes
        if not alphabet:
            # all_phonemes is described in a text file
            self.alphabet = all_phonemes
        else:
            self.alphabet = alphabet  # for testing
        
        # idx to token dictionary starting @ 1
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        # the token @ the zero position = padding token
        self.idx_to_token[0] = pad_token
        
        # reverse dictionary (token to idx)
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        
        # vocabulary size
        self.vocab_size = len(self.alphabet) + 1
        
        # flag adding start & end tokens
        self.add_start_end = add_start_end
        if add_start_end:
            # start token at the position n + 1
            self.start_token_index = len(self.alphabet) + 1
            # start token at the position n + 2
            self.end_token_index = len(self.alphabet) + 2
            # increase vocab size
            self.vocab_size += 2
            # add entries to token to idx & idx to token
            self.idx_to_token[self.start_token_index] = start_token
            self.idx_to_token[self.end_token_index] = end_token
    
    def __call__(self, sentence: str) -> list:
        # create the token sequence based on the token to idx
        sequence = [self.token_to_idx[c] for c in sentence]  # No filtering: text should only contain known chars.
        # start token + sequence + stop token
        if self.add_start_end:
            sequence = [self.start_token_index] + sequence + [self.end_token_index]
        return sequence
    
    # decode from token to phoneme
    def decode(self, sequence: list) -> str:
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


class Phonemizer:
    def __init__(self, language: str, with_stress: bool, njobs=4):
        # phonemizer language
        self.language = language
        # phoonemizer jobs
        self.njobs = njobs
        # phonemizer stress/intonation
        self.with_stress = with_stress
    
    def _filter_string(self, text: str) -> str:
        # return the string convertible in phonemes
        return ''.join([c for c in text if c in all_phonemes])
    
    def filter_characters(self, text: Union[str, list]) -> Union[str, list]:
        # filter function working for strings or lists
        if isinstance(text, list):
            return [self._filter_string(t) for t in text]
        
        elif isinstance(text, str):
            return self._filter_string(text)
        else:
            raise TypeError(f'TextCleaner.clean() input must be list or str, not {type(text)}')
    
    def __call__(self, text: Union[str, list], with_stress=None, njobs=None, language=None)-> Union[str, list]:
        # call function with instance params or arguments (language, njobs,withstress)
        language = language or self.language
        
        njobs = njobs or self.njobs
        
        with_stress = with_stress or self.with_stress
        phonemes = phonemize(text,
                             language=language,
                             backend='espeak',
                             strip=True,
                             preserve_punctuation=True,
                             with_stress=with_stress,
                             njobs=njobs,
                             language_switch='remove-flags')
        
        return self.filter_characters(phonemes)
