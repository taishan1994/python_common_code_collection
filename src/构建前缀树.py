# coding=utf-8
import collections


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self, w):
        '''

        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break
                if current.is_w:
                    result.append([i, j, sentence[i:j + 1]])

        return result


if __name__ == '__main__':
    trie = Trie()
    with open('./data/province.txt', 'r') as fp:
        words = fp.read().strip().split('\n')
        for word in words:
            trie.insert(word)
    sen = '赤壁黄龙大道22号'
    print(trie.get_lexicon(sen))
