import re


def to_latex(s):
    """
    numbers will subscript by default:
    >>> to_latex('x12')
    '$x_{12}$'
    dots will be on top of symbole:
    >>> to_latex('dotx1')
    '$\dot{x_{1}}$'
    """
    res = re.sub(r'(\d+)', r'_{\1}', s)
    res = re.sub(r'(d*)dot(.*)$', r'\\\1dot{\2}', res)
    return f'${res}$'


if __name__ == '__main__':
    import doctest
    doctest.testmod()
