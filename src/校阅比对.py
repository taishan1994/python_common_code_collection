import difflib

def compute_detect_correct_label_list(src_text, trg_text):
    detect_ref_list, correct_ref_list = [], []
    diffs = difflib.SequenceMatcher(None, src_text, trg_text).get_opcodes()
    # print(diffs)
    for (tag, src_i1, src_i2, trg_i1, trg_i2) in diffs:
        if tag == "equal":
            continue

        correct = {
            "src": src_text[src_i1:src_i2],
            "src_start": src_i1,
            "src_end": src_i2,
            "trg": trg_text[trg_i1:trg_i2],
            "trg_start": trg_i1,
            "trg_end": trg_i2,
            "operate": tag,
        }

        correct_ref_list.append(correct)
    return correct_ref_list

def dif(sentence1, sentence2):
    correct_ref_list = compute_detect_correct_label_list(sentence1, sentence2)
    source = []
    target = []
    s_start = 0
    s_end = 0
    t_start = 0
    t_end = 0
    trg_end = None
    src_end = None
    for t in correct_ref_list:
        print(t)
        src_start = t["src_start"]
        src_end = t["src_end"]
        trg_start = t["trg_start"]
        trg_end = t["trg_end"]
        source.append(sentence1[s_start:src_start])
        source.append("【" + sentence1[src_start:src_end] + "】")
        s_start = src_end

        target.append(sentence2[t_start:trg_start])
        target.append("【" + sentence2[trg_start:trg_end] + "】")
        t_start = trg_end

    if src_end <= len(sentence1)-1:
        source.append(sentence1[src_end:])
    if trg_end <= len(sentence2)-1:
        target.append(sentence2[trg_end:])

    return "".join(source), "".join(target)
  
sentence1 = "我欢北精，北京京烤鸭好吃。"
sentence2 = "我喜欢北京，北京烤鸭好吃。"

s1, s2 = dif(sentence1, sentence2)
print(s1)
print(s2)

"""
我【】欢北【精】，北【京】京烤鸭好吃。
我【喜】欢北【京】，北【】京烤鸭好吃。
""
