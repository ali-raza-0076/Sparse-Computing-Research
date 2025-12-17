import csv

a = {}
b = {}

with open('input/matrix_a.csv') as f:
    for row in csv.reader(f):
        i, j, v = int(row[0]), int(row[1]), float(row[2])
        a[(i, j)] = v

with open('input/matrix_b.csv') as f:
    for row in csv.reader(f):
        i, j, v = int(row[0]), int(row[1]), float(row[2])
        b[(i, j)] = v

print(f'A at (24939, 36763): {a.get((24939, 36763), 0)}')
print(f'B at (24939, 36763): {b.get((24939, 36763), 0)}')
print(f'A at (13911, 11038): {a.get((13911, 11038), 0)}')
print(f'B at (13911, 11038): {b.get((13911, 11038), 0)}')

print(f'\nTotal A entries: {len(a)}, File lines: {sum(1 for _ in open("input/matrix_a.csv"))}')
print(f'Total B entries: {len(b)}, File lines: {sum(1 for _ in open("input/matrix_b.csv"))}')
