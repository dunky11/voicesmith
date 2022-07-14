def strip_cont_whitespaces(string: str) -> str:
    new_string = ""
    last_whitespace = False
    for char in string:
        if char == " " and last_whitespace:
            continue
        new_string += char
        last_whitespace = char == " "
    return new_string
