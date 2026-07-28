with open('models/traffic/mog2_parked.py', 'rb') as f:
    data = f.read(200)
print(repr(data))

print("---")

with open('models/traffic/mog2_parked.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if i > 3:
            break
        print(i, repr(line))