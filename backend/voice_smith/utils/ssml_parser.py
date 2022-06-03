from collections import deque
import xml.etree.ElementTree as ET

SUPPORTED_LANGUAGES = ["en-US", "de-DE"]
REQUIRED_ATTRIBUTES = ["voice", "lang"]


def get_voice(node, parent_map):
    current_node = node
    while True:
        if current_node.tag == "voice":
            name = current_node.get("name")
            if name is None:
                raise Exception(
                    r"""
                    You provided a <voice> tag but didnt provide the name attribute, please use the tag like this:
                    <voice name='speaker_name'> your text ... </voice>
                    """
                )
            return {"name": name}
        current_node = parent_map.get(current_node)
        if current_node is None:
            raise Exception(
                r"""
                Your text contained no speaker information, please wrap your text into a 
                <voice name='speaker_name'> your text ... </voice> tag ...
                """
            )


def get_lang(node, parent_map):
    current_node = node
    while True:
        if current_node.tag == "lang":
            lang = current_node.get("{http://www.w3.org/XML/1998/namespace}lang")
            if lang is None:
                raise Exception(
                    r"You provided a <lang> tag but didnt provide the xml:lang attribute, please use the tag like this: "
                    + f"<lang xml:lang='en-US'> your text ... </lang>. Available languages are: {', '.join(SUPPORTED_LANGUAGES)}."
                )
            elif lang not in SUPPORTED_LANGUAGES:
                raise Exception(
                    f"Received a lang tag with an unsopported language: '{lang}'. Please choose from the list of supported languages: {', '.join(SUPPORTED_LANGUAGES)}."
                )
            return {"lang": lang}
        current_node = parent_map.get(current_node)
        if current_node is None:
            raise Exception(
                r"Your text contained no language information, please wrap your text into a "
                + f"<lang xml:lang='en-US'> your text ... </lang> tag ..., supported languages are: {', '.join(SUPPORTED_LANGUAGES)}."
            )


def add_attributes(node, attr_map):
    if node.tag == "voice":
        name = node.get("name")
        if name is None:
            raise Exception(
                r"""
                    You provided a <voice> tag but didnt provide the name attribute, please use the tag like this:
                    <voice name='speaker_name'> your text ... </voice>
                    """
            )
        return {**attr_map, "voice": {"name": name}}
    elif node.tag == "lang":
        lang = node.get("{http://www.w3.org/XML/1998/namespace}lang")
        if lang is None:
            raise Exception(
                r"You provided a <lang> tag but didnt provide the xml:lang attribute, please use the tag like this: "
                + f"<lang xml:lang='en-US'> your text ... </lang>. Available languages are: {', '.join(SUPPORTED_LANGUAGES)}."
            )
        elif lang not in SUPPORTED_LANGUAGES:
            raise Exception(
                f"Received a lang tag with an unsopported language: '{lang}'. Please choose from the list of supported languages: {', '.join(SUPPORTED_LANGUAGES)}."
            )
        return {
            **attr_map,
            "lang": {"{http://www.w3.org/XML/1998/namespace}lang": lang},
        }
    return attr_map


def attributes_from_node(text, attr_map):
    for attr in REQUIRED_ATTRIBUTES:
        if not attr in attr_map:
            raise Exception(
                f"The text '{text}' was not wrapped in a required <{attr}/> tag . "
                + f"Please wrap your text like: <{attr}>{text}</{attr}>"
            )
    return {
        "text": text,
        **attr_map,
    }


def get_non_processed_children(node, nodes_processed_ids):
    non_processed = []
    for child in node:
        if id(child) not in nodes_processed_ids:
            non_processed.append(child)
    return non_processed


def parse_ssml(ssml_string: str):
    root = ET.fromstring(ssml_string)
    assert root.tag == "speak", "Please wrap your SSML in a <speak> tag ..."
    parent_map = {c: p for p in root.iter() for c in p}
    node2attr = {root: add_attributes(root, {})}
    node_stack = deque([root])
    out = []
    nodes_processed_ids = set()
    while len(node_stack) > 0:
        # Stage 1 process current element in stack
        current_node = node_stack.pop()
        attr_map = node2attr[current_node]
        if current_node.text is not None and len(current_node.text.strip()) > 0:
            out.append(attributes_from_node(current_node.text.strip(), attr_map))
        if current_node.tail is not None and len(current_node.tail.strip()) > 0:
            out.append(
                attributes_from_node(
                    current_node.tail.strip(),
                    node2attr[parent_map[current_node]],
                )
            )
        nodes_processed_ids.add(id(current_node))

        # Stage 2 push children of curent node to stack
        children = get_non_processed_children(current_node, nodes_processed_ids)
        for child in children[::-1]:
            node_stack.append(child)
            node2attr[child] = add_attributes(child, attr_map)
    return out


if __name__ == "__main__":
    ssml = """
        <speak>
            <lang xml:lang='en-US'>
                <voice name='michael'>
                    <prosody rate='slow'>This</prosody> 
                    will be 
                </voice>
                <voice name='Tony'>
                    <prosody rate='slow'>a</prosody> 
                    great
                    <lang xml:lang='de-DE'>
                        day!
                    </lang>
                </voice>
                <voice name='Test'>
                    And a test
                </voice>
            </lang>
        </speak>
    """
    attr = parse_ssml(ssml)
    print(attr)
