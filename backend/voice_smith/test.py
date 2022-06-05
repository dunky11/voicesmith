args = {"a": 1, "b": 2, "c": 3}


def test(a, b, **kwargs):
    print(a)
    print(b)


test(**args)
