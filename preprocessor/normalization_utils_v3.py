from mat2vec_origin_processing.process import MaterialsTextProcessor
from normalization_utils_helper import title_keyword, abstract_keyword, html_mappings, matsciBERT_mapping, \
                                        CONTROLS, HYPHENS, DOUBLE_QUOTES, ACCENTS, SINGLE_QUOTES, APOSTROPHES, SLASHES, TILDES, MINUSES, \
                                        CHAR_REPLACEMENTS
import unicodedata

class Chain(object):
    def __init__(self, *callables):
        self.callables = callables

    def __call__(self, value):
        for func in self.callables:
            value = func(value)
        return value
    
class Battery2Vec_Processor:

    _title_keyword = []
    _abstract_keyword = []
    _html_mappings = {}
    _matscibert_mapping = {}
    _substitutor_mapping = []

    text_process = None
    normalize = None
    normalize2 = None

    def __init__(self):
        self._title_keyword = title_keyword + [key.upper() for key in title_keyword]
        self._abstract_keyword = sorted(abstract_keyword, key=len, reverse=True)
        self._html_mappings = html_mappings
        self._html_mappings.update({k.replace("&", "& ").replace(";", " ;"):v for k,v in html_mappings.items()})
        self._matscibert_mapping = matsciBERT_mapping
        self._substitutor_mapping = CHAR_REPLACEMENTS

        self.text_process = MaterialsTextProcessor()
        self.normalize = Chain(
            self._02_remove_copyright_sentence,
            self._03_remove_parenthetical_keyword
        )
        self.normalize2 = Chain(
            self._05_text_NFKC,
            self._06_html_mapping,
            self._07_matscibert_mapping,
            self._08_connect_splecific_sentences,
            self._09_strict_normalize,
            self._10_substitutor_normalize
        )

    def _01_remove_paper(self, _title:str) -> bool:
        """ if True, remove this paper. """
        for key in self._title_keyword:
            if _title.find(key) != -1: return True
        return False

    def _02_remove_copyright_sentence(self, _text:str) -> str:
        copyright_idx = _text.find("copyright")
        while copyright_idx != -1:
            copyright_endidx = _text.find(".", copyright_idx)
            copyright_endidx = len(_text) if copyright_endidx == -1 else copyright_endidx
            copyright_startidx = _text.rfind(".", 0, copyright_idx)
            copyright_startidx = 0 if copyright_startidx == -1 else copyright_startidx

            _text = _text[:copyright_startidx] + _text[copyright_endidx:]
            copyright_idx = _text.find("copyright")
        return _text

    def _03_remove_parenthetical_keyword(self, _text:str) -> str:
        for key in self._abstract_keyword:
            _text = _text.replace(key, "")
        try: 
            if _text[-1] == "<": _text = _text[:-2].strip()
        except: pass
        else: _text = _text.strip()

        return _text
    
    def _04_remove_under_bar(self, _text:str, _doi="") -> str:
        if _doi is None: return _text
        if _doi.find("10.1007/") != -1:
            _text = _text.replace("_", "")
        return _text
    
    def _05_text_NFKC(self, _text:str) -> str:
        """each char"""
        _st = ""
        for _t in _text:
            _st += unicodedata.normalize("NFKC", _t)
        return _st
    
    def _06_html_mapping(self, _text:str, version=3) -> str:
        if version in [1, 2]:
            _text = _text.replace("&gt;", "")
            _text = _text.replace("&lt;", "")
            _text = _text.replace("&amp;", "")
        elif version == 3:
            for k, v in self._html_mappings.items():
                _text = _text.replace(k, v)
        return _text
    
    def _07_matscibert_mapping(self, _text:str) -> str:
        for k, v in self._matscibert_mapping.items():
            _text = _text.replace(k, v)
        return _text

    def _08_connect_splecific_sentences(self, _text:str) -> str:
        stmt = self.text_process.tokenize(_text, split_oxidation=True, keep_sentences=False)
        tmp_stmt = []
        prev_flg = 0
        for i, s in enumerate(stmt):
            s = s.strip()
            cur_flg = 0

            try:
                if 49 <= ord(s[-1]) <= 57: tmp_s = self.text_process.normalized_formula(s) # 1~9
                else: tmp_s = s
                if len(tmp_s) > 1:
                    if tmp_s[-2:] in ["-x", "-y", "-z", "−x", "−y", "−z", "-m", "-n", "-t", "−m", "−n", "−t"]: cur_flg = 1
                else:
                    if tmp_s in ["x", "y", "z", "m", "n", "t"]: cur_flg = 1
                if self.text_process.is_simple_formula(tmp_s): cur_flg = 1
                if (self.text_process.is_element(tmp_s)) and not (i == 0 and tmp_s == "In"): cur_flg = 1
            except:
                tmp_s = s
                cur_flg = 0

            if prev_flg == 1 and cur_flg == 1: tmp_stmt[-1] += s
            else: tmp_stmt.append(s)
            prev_flg = cur_flg
        _st = " ".join(tmp_stmt)
        return _st

    def _09_strict_normalize(self, _text:str) -> str:
        def _09_01_controls(_st:str) -> str:
            for control in CONTROLS:
                _st = _st.replace(control, '')
            return _st
        def _09_02_unusal_whitespace(_st:str) -> str:
            _st = _st.replace('\u000b', ' ').replace('\u000c', ' ').replace(u'\u0085', ' ')
            _st = _st.replace('\u2028', '\n').replace('\u2029', '\n').replace('\r\n', '\n').replace('\r', '\n')
            return _st
        def _09_03_hyphens(_st:str) -> str:
            for hyphen in HYPHENS | MINUSES:
                _st = _st.replace(hyphen, '-')
            _st = _st.replace('\u00ad', '-')
            return _st
        def _09_04_quotes(_st:str) -> str:
            for double_quote in DOUBLE_QUOTES:
                _st = _st.replace(double_quote, '"')  # \u0022
            for single_quote in (SINGLE_QUOTES | APOSTROPHES | ACCENTS):
                _st = _st.replace(single_quote, "'")  # \u0027
            _st = _st.replace('′', "'")     # \u2032 prime
            _st = _st.replace('‵', "'")     # \u2035 reversed prime
            _st = _st.replace('″', "''")    # \u2033 double prime
            _st = _st.replace('‶', "''")    # \u2036 reversed double prime
            _st = _st.replace('‴', "'''")   # \u2034 triple prime
            _st = _st.replace('‷', "'''")   # \u2037 reversed triple prime
            _st = _st.replace('⁗', "''''")  # \u2057 quadruple prime
            return _st
        def _09_05_ellipsis(_st:str) -> str:
            _st = _st.replace('…', '...').replace(' . . . ', ' ... ')  # \u2026
            return _st
        def _09_06_slashes(_st:str) -> str:
            for slash in SLASHES:
                _st = _st.replace(slash, '/')
            return _st
        def _09_07_tildes(_st:str) -> str:
            for tilde in TILDES:
                _st = _st.replace(tilde, "~")
            return _st
        
        _text = _09_01_controls(_text)
        _text = _09_02_unusal_whitespace(_text)
        _text = _09_03_hyphens(_text)
        _text = _09_04_quotes(_text)
        _text = _09_05_ellipsis(_text)
        _text = _09_06_slashes(_text)
        _text = _09_07_tildes(_text)
        _text = ' '.join(_text.strip().split())
        return _text

    def _10_substitutor_normalize(self, _text:str) -> str:
        for pattern, replacement in self._substitutor_mapping:
            _text = _text.replace(pattern, replacement)
        return _text
    
    def _11_16_mat2vec_normalize(self, _text:str, is_version_improving=True):
        _text, _mat_list = self.text_process.process(_text, is_version_improving=is_version_improving)
        return _text, _mat_list
    
    def process(self, ttl, abst, doi):
        ttl_mat_list  = []
        abst_mat_list = []

        if ttl is not None:
            remove_flg = self._01_remove_paper(ttl)
            if remove_flg:
                return "", []
            
            ttl = self.normalize(ttl)
            ttl = self._04_remove_under_bar(ttl, doi)
            ttl = self.normalize2(ttl)
            ttl, ttl_mat_list = self._11_16_mat2vec_normalize(ttl)
        else: ttl = []

        if abst is not None:
            abst = self.normalize(abst)
            abst = self._04_remove_under_bar(abst, doi)
            abst = self.normalize2(abst)
            abst, abst_mat_list = self._11_16_mat2vec_normalize(abst)
        else: abst = []

        corpus = ' '.join(ttl + abst) # list + list -> str
        formulas = ttl_mat_list + abst_mat_list # list + list -> list

        return corpus, formulas
    
    def process_for_one_text(self, _text, version=3):
        
        if version == 1:
            """mat2vec re-production version"""
            _text = self.normalize(_text)
            corpus, formulas = self._11_16_mat2vec_normalize(_text, is_version_improving=False)
        
        elif version == 2:
            """improving first version"""
            _text = self._06_html_mapping(_text, version=version)
            _text = self._08_connect_splecific_sentences(_text)
            _text = self._02_remove_copyright_sentence(_text)
            _text = self._03_remove_parenthetical_keyword(_text)
            # _text = self._04_remove_under_bar(_text)
            corpus, formulas = self._11_16_mat2vec_normalize(_text, is_version_improving=False)

        elif version == 3:
            """more improving version"""
            _text = self.normalize(_text)
            _text = self.normalize2(_text)
            corpus, formulas = self._11_16_mat2vec_normalize(_text, is_version_improving=True)
        
        else:
            print(f"version error : normalization_utils.py version : {version}")

        return ' '.join(corpus), formulas

def main(version):
    tmp_text1 = "hello world. &lt;"
    # result ->  hello world . <
    tmp_text2 = "This sentence is test sentence which has no phrase."
    # result ->  this sentence is test sentence which has no phrase .
    tmp_text3 = "The gravimetric capacity of the battery is 1000 mAh/g and current density jc is 1.0 mA/cm2."
    # result ->  the gravimetric capacity of the battery is <nUm> mAh / g and current density jc is <nUm> mA / cm2 .

    b2vp = Battery2Vec_Processor()
    print(b2vp.process_for_one_text(tmp_text1, version)[0])
    print(b2vp.process_for_one_text(tmp_text2, version)[0])
    print(b2vp.process_for_one_text(tmp_text3, version)[0])
    

if __name__ == "__main__":
    import sys
    setting_version = int(sys.argv[1])
    main(setting_version)