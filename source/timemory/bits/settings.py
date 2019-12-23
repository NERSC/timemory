#!/usr/bin/env python

import re

f = open('settings.txt', 'r')
last = None


class accessor():

    def __init__(self, line):
        self.fields = re.split(r"[\(\),]+", line)[1:]
        for i in range(len(self.fields)):
            self.fields[i] = self.fields[i].strip()

    def __str__(self):
        return "_TRY_CATCH_NVP({}, {})".format(self.fields[2], self.fields[1])


accessor_list = []

while True:
    line = f.readline()
    if len(line) == 0:
        break
    else:
        line = line.strip()
        if line.find("///") == 0:
            last = line
        else:
            # if last is not None:
                # print("{}".format(last))
                # print("{}".format(line))
            accessor_list.append(accessor(line))
            last = None

for field in accessor_list:
    if field.fields[2] == ";":
        continue
    print("{}".format(field))
