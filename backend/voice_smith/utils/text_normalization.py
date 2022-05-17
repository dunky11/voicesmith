import re
import inflect
import re
import unicodedata

def remove_cont_whitespaces(text):
    new_string = ""
    last_char = None
    for char in text:
        if char == " " and last_char == " ":
            continue
        last_char = char
        new_string += char
    return new_string


class NumberNormalizer():
    """ FROM https://github.com/keithito/tacotron/blob/master/text/numbers.py
    """
    inflector = inflect.engine()
    commanumber_re = re.compile(r'([0-9][0-9\,]+[0-9])')
    decimalnumber_re = re.compile(r'([0-9]+\.[0-9]+)')
    pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
    dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
    ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
    number_re = re.compile(r'[0-9]+')

    def remove_commas(self, m):
        return m.group(1).replace(',', '')

    def expand_decimal_point(self, m):
        return m.group(1).replace('.', ' point ')

    def expand_dollars(self, m):
        match = m.group(1)
        parts = match.split('.')
        if len(parts) > 2:
            return match + ' dollars'    # Unexpected format
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = 'dollar' if dollars == 1 else 'dollars'
            cent_unit = 'cent' if cents == 1 else 'cents'
            return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
        elif dollars:
            dollar_unit = 'dollar' if dollars == 1 else 'dollars'
            return '%s %s' % (dollars, dollar_unit)
        elif cents:
            cent_unit = 'cent' if cents == 1 else 'cents'
            return '%s %s' % (cents, cent_unit)
        else:
            return 'zero dollars'


    def expand_ordinal(self, m):
        return self.inflector.number_to_words(m.group(0))


    def expand_number(self, m):
        num = int(m.group(0))
        if num > 1000 and num < 3000:
            if num == 2000:
                return 'two thousand'
            elif num > 2000 and num < 2010:
                return 'two thousand ' + self.inflector.number_to_words(num % 100)
            elif num % 100 == 0:
                return self.inflector.number_to_words(num // 100) + ' hundred'
            else:
                return self.inflector.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
        else:
            return self.inflector.number_to_words(num, andword='')


    def __call__(self, text: str) -> str:
        text = re.sub(self.commanumber_re, self.remove_commas, text)
        text = re.sub(self.pounds_re, r'\1 pounds', text)
        text = re.sub(self.dollars_re, self.expand_dollars, text)
        text = re.sub(self.decimalnumber_re, self.expand_decimal_point, text)
        text = re.sub(self.ordinal_re, self.expand_ordinal, text)
        text = re.sub(self.number_re, self.expand_number, text)
        return text

class EnglishTextNormalizer():
    """ FROM https://github.com/Kyubyong/g2p/blob/master/g2p_en/g2p.py
    """ 
    def __init__(self):
        self.number_normalizer = NumberNormalizer()


    def __call__(self, text: str) -> str:
        text = self.number_normalizer(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                        if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")
        return text