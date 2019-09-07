import sys
import random

if __name__ == '__main__':
    train_text_file = sys.argv[1]
    train_label_file = sys.argv[2]
    test_text_file = sys.argv[3]
    test_label_file = sys.argv[4]
    name = sys.argv[5]
    with open(train_label_file) as f:
        train_label_lineList = [line.rstrip('\n') for line in f]

    with open(train_text_file) as f:
        train_text_lineList = [line.rstrip('\n') for line in f]

    with open(test_text_file) as f:
        test_text_lineList = [line.rstrip('\n') for line in f]

    with open(test_label_file) as f:
        test_label_lineList = [line.rstrip('\n') for line in f]

    data = []
    for i in range(len(train_label_lineList)):
        data.append((train_label_lineList[i], train_text_lineList[i]))

    for i in range(len(test_label_lineList)):
        data.append((test_label_lineList[i], test_text_lineList[i]))


    random.shuffle(data)



    with open(name + '.train', "w+") as f:
        for i in range(int(len(data) * 0.9)):
            print(i)
            f.write('__label__' + str(int(data[i][0]) + 1) + ' , ' + data[i][1] + '\n')
        print(i)

    with open(name + '.test', "w+") as f:
        for i in range(int(len(data) * 0.9),len(data)):
            print(i)
            f.write('__label__' + str(int(data[i][0]) + 1) + ' , ' + data[i][1] + '\n')
        print(i)

