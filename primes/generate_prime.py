from sympy import randprime, isprime


def gen_semiprime(bits_N=512):
    bits_p = bits_N // 2
    lo = 1 << (bits_p - 1)
    hi = (1 << bits_p) - 1

    while True:
        p = randprime(lo, hi)
        if isprime(p):
            break

    while True:
        q = randprime(lo, hi)
        if isprime(q) and q != p:
            break

    N = p * q
    return p, q, N

for i in range(10):
    p, q, N = gen_semiprime(64)
    print("p bits:", p.bit_length(), "p:", p)
    print("q bits:", q.bit_length(), "q:", q)
    print("N bits:", N.bit_length(), "N:", N)
    print()