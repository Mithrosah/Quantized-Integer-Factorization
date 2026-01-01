# Quantized-Integer-Factorization
Solving integer factorization using Quantized Ising Machine.

optimize by success rate (default):
```sh
python solve.py --metric success
```

optimize by median time-to-first-hit:
```sh
python solve.py --metric ttfh
```

optimize by mean time-to-first-hit
```sh
python solve.py --metric ttfh --ttfh_stat mean
```