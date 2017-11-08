# Author: Rocco Haro
# Used for debugging purposes

# How to incorporate into your code:
# 1 import displayClass as dC
# . . .
# 40 currLocalFiles = list(locals().iteritems())
# 41 ex1 = "1"
# 42 lux2 = 189.032
# 43 d = dC.displayFuncs(currLocalFiles)
# 44 d.display(ex1, lux2)

# references:
# https://www.accelebrate.com/blog/using-defaultdict-python/
# https://stackoverflow.com/questions/4381569/python-os-module-open-file-above-current-directory-with-relative-path
# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
# https://stackoverflow.com/questions/38230322/print-name-of-variable-not-its-value-in-python
# https://stackoverflow.com/questions/3405073/generating-dictionary-keys-on-the-fly

# Tips:
# declare this in every function you plan on working:
        # def getLocalVars(self):
        #     """ returns all the local variables, including self """
        #     return list(locals().iteritems())
# then you can allocate d in line 43 of the example as = dc.displayFuncs(getLocalVars())
# This will enable you to use the display() with less clutter

from collections import defaultdict

class displayFuncs:
    def __init__(self, *args):
        self.args = args
        self.objectsInClass = self.args[0]

    def display(self, *targets):
        """ Prints all of the variables passed in with the format:
            <varName> , <varValue>
            * e.g. if x = 1, then display(x) -> <'x'>, <1>
            * can accept any number, and type of targets.
        """
        dicts = defaultdict(lambda: defaultdict(dict))

        for name, value in self.objectsInClass:
            if value in targets:
                dicts[name] = value
        for key, value in dicts.items():
            print(key,value)

    def updateVarInfolist(self, freshVariables):
        print("updating",freshVariables)
        self.objectsInClass = freshVariables
        x= 3
