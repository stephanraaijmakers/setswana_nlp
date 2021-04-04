import sys
import re
from math import floor

if __name__=="__main__":
    with open(sys.argv[1],"r") as fp:
        lines=fp.readlines()
    for line in lines:
        m=re.match("^\d+\t+\d*\.?\s*(.+)$",line.rstrip())
        if m:            
            print("%s"%(m.group(1)))
        else:
            print("Error:%s"%(line))
            exit(0)
    fp.close()
    exit(0)
                  
            
