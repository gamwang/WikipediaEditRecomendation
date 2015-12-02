import json
from sets import Set

def clean_user_contrib(filename):
    outfile = filename.split(".")[0] + ".out"
    with open(filename, 'r') as f, open(outfile, 'a+') as o:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            user_histories = json.loads(line)
            if ("query" in user_histories.keys() and
            len(user_histories["query"]["usercontribs"]) > 0):
                user_id = user_histories["query"]["usercontribs"][0]["userid"]
                user_contrib = user_histories["query"]["usercontribs"]
                contrib = clean_helper(user_contrib)
                if len(contrib) > 2:
                    contrib = (user_id, contrib)
                    o.write(str(contrib))
                    o.write('\n')

def clean_helper(user_contrib):
    contrib = map(lambda x: x['title'], user_contrib)
    contrib = list(Set(contrib))
    contrib = filter(lambda x: "Talk:" not in x, contrib)
    contrib = filter(lambda x: "File:" not in x, contrib)
    contrib = filter(lambda x: "Draft:" not in x, contrib)
    contrib = filter(lambda x: "Category:" not in x, contrib)
    contrib = filter(lambda x: "User:" not in x, contrib)
    contrib = filter(lambda x: "Template:" not in x, contrib)
    contrib = filter(lambda x: "Wikipedia:" not in x, contrib)
    contrib = filter(lambda x: "talk:" not in x, contrib)
    contrib = filter(lambda x: "Portal:" not in x, contrib)
    contrib = filter(lambda x: "Help:" not in x, contrib)
    return contrib



clean_user_contrib("0.txt")
clean_user_contrib("1.txt")
clean_user_contrib("2.txt")
clean_user_contrib("3.txt")
clean_user_contrib("4.txt")
clean_user_contrib("5.txt")
clean_user_contrib("6.txt")
clean_user_contrib("7.txt")
clean_user_contrib("8.txt")
clean_user_contrib("9.txt")
