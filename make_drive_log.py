import sys, glob, csv

folder = sys.argv[1]
steer = float(sys.argv[2])

with open('training_data\\{}\\driving_log.csv'.format(folder), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in glob.glob('training_data\\{}\\IMG\\*'.format(folder)):
        print(i,steer)
        data = [i,i,i,steer,0,0,15]
        writer.writerow(data)
