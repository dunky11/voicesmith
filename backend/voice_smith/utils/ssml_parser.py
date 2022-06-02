from lxml import etree


def parse_ssml(ssml_string: str):
    string = etree.fromstring(ssml_string)
    assert string.tag == "speak", "Please wrap your SSML in a <speak> tag ..."
    current_speaker = None


def search_tree(node, speakers, words, ):
    


if __name__ == "__main__":
    """ssml_string = (
        "<speak>Today <prosody rate='slow'>will be</prosody> a <lang></lang></speak>"
    )"""
    test = []

    def app(le):
        le.append("a")

    app(test)
    print(test)
