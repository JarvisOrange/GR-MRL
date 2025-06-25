import numpy as np
import datetime

data_dict = {
    "B": '2017-01-01',
    "L": '2012-05-01',
    'C': '2018-01-01',
    'S': '2018-01-01',
}

dataset_name_dict = {
    'B':0,
     'L':1,
     'C':2,
     'S':3,
}

city = 'S'





def func(city):
    data = np.load('./Raw_Data/'+city+'/dataset_expand.npy')
    date = datetime.datetime.strptime(data_dict[city], '%Y-%m-%d').date()
    weekday_start = date.weekday()
    print(weekday_start,data.shape)
    z=0
    if city == 'C' or city == 'S':
        temp = np.zeros((data.shape[0]*2-1, data.shape[1], 7), dtype=np.float32)
        for i in range(0, data.shape[0]):
            for j in range(data.shape[1]):
                if i == data.shape[0] - 1:
                    temp[i*2, j, 0] = data[i, j, 0] #speed
                    temp[2*i, j, 1] = z #index_time
                    temp[2*i, j, 2] = (weekday_start * 288 + z) % 2016
                    temp[2*i, j, 3] = j

                    s = dataset_name_dict[city]

                    temp[2*i, j, 4] = s

                    temp[2*i, j, 5] = cfg['dataset_info'][city]['mean']
                    temp[2*i, j, 6] = cfg['dataset_info'][city]['std']
                    
                else:
                    temp[i*2, j, 0] = data[i, j, 0] #speed
                    temp[2*i+1, j, 0] = (data[i, j, 0] + data[i+1, j, 0]) / 2 #speed
                    
                    temp[2*i, j, 1] = z #index_time
                    temp[2*i+1, j, 1] = z+1 #index_time

                    temp[2*i, j, 2] = (weekday_start * 288 + z) % 2016
                    temp[2*i+1, j, 2] = (weekday_start * 288 + z + 1) % 2016


                    temp[2*i, j, 3] = j
                    temp[2*i+1, j, 3] = j

                    s = dataset_name_dict[city]

                    temp[2*i, j, 4] = s
                    temp[2*i+1, j, 4] = s

                    temp[2*i, j, 5] = cfg['dataset_info'][city]['mean']
                    temp[2*i+1, j, 5] = cfg['dataset_info'][city]['mean']

                    temp[2*i, j, 6] = cfg['dataset_info'][city]['std']
                    temp[2*i+1, j, 6] = cfg['dataset_info'][city]['std']
                
            z += 2
        np.save('./Raw_Data/'+city+'/dataset_new.npy', temp)
    else:
        temp = np.zeros((data.shape[0], data.shape[1], 7), dtype=np.float32)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                temp[i, j, 0] = data[i, j, 0] #speed
                temp[i, j, 1] = data[i, j, 2] #index_time
                
                t = (weekday_start * 288 + z) % 2016
                temp[i, j, 2] = t
                temp[i, j, 3] = j
                s = dataset_name_dict[city]
                temp[i, j, 4] = s


                temp[i, j, 5] = cfg['dataset_info'][city]['mean']
                temp[i, j, 6] = cfg['dataset_info'][city]['std']
            z += 1
        np.save('./Raw_Data/'+city+'/dataset_new.npy', temp)

    data = np.load('./Raw_Data/'+city+'/dataset_new.npy')
    print(data.shape)
    print(data[1,3,:])
func('B')
