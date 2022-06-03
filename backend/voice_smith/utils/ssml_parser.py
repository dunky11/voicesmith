from collections import deque
import xml.etree.ElementTree as ET

SUPPORTED_LANGUAGES = ["en-US"]


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


def attributes_from_node(text, parent_node, parent_map):
    return {
        "text": text,
        "voice": get_voice(parent_node, parent_map),
        "lang": get_lang(parent_node, parent_map),
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
    node_stack = deque([root])
    attributes = []
    nodes_processed_ids = set()
    while len(node_stack) > 0:
        # Stage 1 process current element in stack
        current_node = node_stack.pop()
        if current_node.text is not None and len(current_node.text.strip()) > 0:
            attributes.append(
                attributes_from_node(
                    current_node.text.strip(), current_node, parent_map
                )
            )
        if current_node.tail is not None and len(current_node.tail.strip()) > 0:
            attributes.append(
                attributes_from_node(
                    current_node.tail.strip(), parent_map[current_node], parent_map
                )
            )
        nodes_processed_ids.add(id(current_node))

        # Stage 2 push children of curent node to stack
        children = get_non_processed_children(current_node, nodes_processed_ids)
        if len(children) > 0:
            node_stack.extend(children)
    return attributes


if __name__ == "__main__":
    ssml = """
        <speak>
            <lang xml:lang='en-US'>
                <voice name='michael'>
                    <prosody rate='slow'>will be</prosody> 
                    a 
                </voice>
            </lang>
        </speak>
    """
    attr = parse_ssml(ssml)
    print(attr)
