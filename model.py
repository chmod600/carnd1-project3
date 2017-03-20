import csv
base_path = "/Users/Akshay/Desktop/p3/data/"
lines = []
with open(base_path + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = base_path + "IMG/" + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
