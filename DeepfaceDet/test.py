from sklearn.metrics import f1_score


# y_true = [0, 1, 0, 0]
# y_pred = [0, 0, 0, 0] 

y_true = [1, 1, 1, 1, 2, 2, 2]
y_pred = [1, 1, 2, 1, 2, 2, 1]

print(f1_score(y_true, y_pred))