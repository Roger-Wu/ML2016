import sys
import numpy as np

def main():
    colIdx = int(sys.argv[1])
    inFileName = sys.argv[2]
    outFileName = 'ans1.txt'

    mat = np.loadtxt(inFileName)
    col = mat[:, colIdx]
    col.sort()

    with open(outFileName, 'w') as outFile:
        print(','.join(map(str, col.tolist())), file=outFile)
        # outFile.write(','.join(map(str, col.tolist())))

if __name__ == '__main__':
    main()
