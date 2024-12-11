L = ["0002", "0006", "0005"]
text = "0002 0006 0005 0006 0005 0004"
sum = 0
for l in L:
    print(l)
    sum += text.count(l)
print("sum", sum)