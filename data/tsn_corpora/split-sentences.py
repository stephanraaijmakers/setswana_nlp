import sys
import re
from math import floor

if __name__=="__main__":
    with open(sys.argv[1],"r") as fp:
        lines=fp.readlines()

    for line in lines:
        m=re.match("^\d+\t+\d*\.?\s*(.+)$",line.rstrip())
        if m:            
            words=m.group(1).split(" ")
            nb_words=len(words)
            half=int(floor(nb_words/2.0))
            first_half=re.sub("^\(\d+\)\s+","",' '.join(words[:half]))
            second_half=' '.join(words[half:])
            
            print("%s.\n%s"%(first_half,second_half))
        else:
            print("Error:%s"%(line))

    fp.close()
    exit(0)
                  
            
