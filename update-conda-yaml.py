#!/usr/bin/env python

import os
import sys
import traceback
import subprocess as sp

def read_file(fname):
    content = []
    with open(fname) as f:
        for line in f:
            content.append(line)
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip("\n") for x in content]
    return content

def main(new_git_rev):
    content = read_file("recipe/meta.yaml")

    fm = open("recipe/meta.yaml", "w")
    old_git_rev = ""
    for l in content:
        lstrip = l.strip().strip("-").strip("\t")
        entries = lstrip.split()
        if len(entries) > 1:
            if entries[0] == "git_rev:":
                old_git_rev = entries[1]
                l = l.replace(old_git_rev, new_git_rev)
        fm.write("{}\n".format(l))
        print("{}".format(l))

    fm.close()
    print("\nOld git revision: {}".format(old_git_rev))
    print("\nNew git revision: {}\n".format(new_git_rev))


if __name__ == "__main__":
    try:

        git_rev = ""
        if len(sys.argv) < 2:
            process = sp.Popen(['git', 'rev-parse', 'HEAD'], stdout=sp.PIPE)
            out, err = process.communicate()
            git_rev = "{}".format(out.decode("utf-8").strip())
            #git_rev = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
        else:
            git_rev = sys.argv[1]

        main(git_rev)

    except Exception as e:
        print ('Error running pyctest - {}'.format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)

    sys.exit(0)
