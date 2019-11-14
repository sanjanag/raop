def read_moral_dic():
    f = open("./resources/MoralFoundations.dic", "r")
    lines = f.readlines()
    return get_moral_dic(lines)


def get_moral_dic(lines):
    dic = {}
    for line in lines:
        line = line
        line = line.replace("%", "")
        line = line.split()
        if len(line) > 0:
            if not line[0].isnumeric():
                word = line[0]
                if word[-1] == "*":
                    word = word[:-1] + ".*"
                categories = line[1:]
                for category in categories:
                    if category not in dic:
                        dic[category] = []
                    dic[category].append(word)
    return dic
