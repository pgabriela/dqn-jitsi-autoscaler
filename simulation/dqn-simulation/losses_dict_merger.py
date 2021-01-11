import pickle


losses_dict = pickle.load(open('r2/losses_dict.pkl', 'rb'))
idx = 3

while idx < 15:
    try:
        temp = pickle.load(open('r' + str(idx) + '/losses_dict.pkl', 'rb'))
    except:
        idx += 1
        continue
    for p in temp:
        for tj in temp[p]:
            for ij in temp[p][tj]:
                (
                    losses_dict
                    .setdefault(p, {})
                    .setdefault(tj, {})
                )[ij] = temp[p][tj][ij]
    idx += 1

pickle.dump(losses_dict, open('losses_dict.pkl', 'wb'))
