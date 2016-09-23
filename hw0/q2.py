import sys
from PIL import Image

def main():
    inFileName = sys.argv[1]
    outFileName = 'ans2.png'

    im = Image.open(inFileName)
    im = im.rotate(180)
    im.save(outFileName)

if __name__ == '__main__':
    main()
