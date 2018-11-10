FILE_NAME = 'this_thing.html'
TRAINING_SET_OUTPUT_DIR = 'training_data'

header = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
'''

tail = '''
</body>
</html>
'''

def preProcessRows():
    with open(FILE_NAME, 'r') as f:
        l = []
        for cnt, line in enumerate(f):
            l.append(line)
            if cnt % 10 == 0 and cnt != 0:
                with open(TRAINING_SET_OUTPUT_DIR + "/tset"+str(cnt//10)+".html", 'w') as f2:
                    f2.write(header)
                    f2.writelines(l)
                    f2.write(tail)
                l = []


if __name__ == '__main__':
    preProcessRows()
