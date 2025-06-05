import numpy as np
import datetime

data_dict = {
    "BAY": '2017-01-01',
    "LA": '2012-05-01',
    'CD': '2018-01-01',
    'SZ': '2018-01-01',
}
city = 'SZ'
data = np.load('./raw_data/'+city+'/dataset_expand.npy')
print(data[1,3,:])

date = datetime.datetime.strptime(data_dict[city], '%Y-%m-%d').date()
weekday_start = date.weekday()
print(weekday_start,data.shape)
exit(0)
z=0
temp = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.float32)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        temp[i, j, 0] = data[i, j, 0]
        temp[i, j, 1] = data[i, j, 1]
        temp[i, j, 2] = z
        temp[i, j, 3] = (weekday_start * 144 + z) % 1008
    z += 1
np.save('./raw_data/'+city+'/dataset_expand.npy', temp)