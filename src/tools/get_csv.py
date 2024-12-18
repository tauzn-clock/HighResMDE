import csv

TRAIN_CSV = "/HighResMDE/nddepth_custom/nyudepthv2_train_files_with_gt_dense.txt"
TEST_CSV = "/HighResMDE/nddepth_custom/nyudepthv2_test_files_with_gt.txt"

def get_csv(csv_file):

    data = []

    focal = 518.8579
    fx = focal
    fy = focal
    cx = 325.5824
    cy = 253.7362
    depth_scale = 1000
    depth_max = 10

    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            new_entry = [row[0], row[1], fx, fy, cx, cy, depth_scale, depth_max]
            data.append(new_entry)

    return data

train_arr = get_csv(TRAIN_CSV)
test_arr = get_csv(TEST_CSV)

with open("train.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerows(train_arr)

with open("test.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerows(test_arr)
    
print(len(train_arr))
print(len(test_arr))