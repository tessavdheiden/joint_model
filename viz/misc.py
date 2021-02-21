def to_latex(x):
    s = str(x)
    for i,c in enumerate(s):
        if c.isdigit():
            s = s[:i] + '_' + s[i:]
            break

    K = ['dddot', 'ddot', 'dot']
    V = ['\dddot{', '\ddot{', '\dot{']
    d = dict(zip(K, V))
    for k in K:
        if k in s:
            s = s.replace(k, d[k]) + "}"
            break

    return f'${s}$'