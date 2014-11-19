# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:05:25 2014

@author: justinmaojones
"""
import re

w = "1grade\xe2"

if re.search('[^a-zA-Z0-9_]',w):
    print w,"not okay"
else:
    print w,"is y"

for c in w:
    if re.search('[a-zA-Z0-9_]',c):
        print c,"True"
    else:
        print c,"False"
        w = str.replace(w,c," ")

print w


def RemoveSpecialUnicode(words):
    for c in words:
        if not re.search('[a-zA-Z0-9_]',c):
            words = str.replace(words,c," ")
    return words

essay = "My exceptional students are reluctant to read and their reading \xe2\x80\x9cEnd of Grade\xe2\x80\x9d (EOG) scores are Level I and Level II.  I am responsible for providing instruction in reading and remediation to prepare them for state EOG tests.  I work with students in g don't"
#print unicode(essay)
revessay = RemoveSpecialUnicode(essay)