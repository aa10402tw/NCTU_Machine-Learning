
def read_data(file_name):
    datas = []
    with open(file_name) as f:
        for line in f:
            data = line.split(',')
            data[0] = float(data[0])
            data[1] = float(data[1])
            datas.append(data)
    return datas