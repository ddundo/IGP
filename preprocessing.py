import os
import numpy as np
import pandas as pd
import datetime

path = './gcmt_data/'

date, lat, lon, depth, mb, MS, loc_name, CMT_name, exp, Mrr, Mrr_std, Mtt, Mtt_std, Mpp, Mpp_std, Mrt, Mrt_std, Mrp, Mrp_std, Mtp, Mtp_std = ([] for _ in range(21))

for filename in os.listdir(path):
    print('Reading', filename)
    print(len(lat))
    i = 1

    with open(path + filename) as f:
        for line in f:

            if i % 5 == 1:

                if line[0] == ' ':
                    next(f)  #Skip event
                    next(f)
                    next(f)
                    next(f)
                    continue

                s = line.split(None, 8)
                if len(s) != 9:
                    next(f)  #Skip event
                    next(f)
                    next(f)
                    next(f)
                    continue

                date.append(s[1])
                lat.append(float(s[3]))
                lon.append(float(s[4]))
                depth.append(float(s[5]))
                mb.append(float(s[6]))
                MS.append(float(s[7]))
                loc_name.append(s[8].rstrip())

            if i % 5 == 2:
                s = line.split()
                CMT_name.append(s[0])

            elif i % 5 == 4:
                s = line.split()
                exp.append(int(s[0]))
                Mrr.append(float(s[1]))
                Mrr_std.append(float(s[2]))
                Mtt.append(float(s[3]))
                Mtt_std.append(float(s[4]))
                Mpp.append(float(s[5]))
                Mpp_std.append(float(s[6]))
                Mrt.append(float(s[7]))
                Mrt_std.append(float(s[8]))
                Mrp.append(float(s[9]))
                Mrp_std.append(float(s[10]))
                Mtp.append(float(s[11]))
                Mtp_std.append(float(s[12].rstrip()))

            i += 1


d = {'date' : date, 'lat' : lat, 'lon' : lon, 'depth' : depth, 'mb' : mb, 'MS' : MS, 'loc_name' : loc_name, 'CMT_name' : CMT_name, 'exp' : exp, 'Mrr' : Mrr, 'Mrr_std' : Mrr_std, 'Mtt' : Mtt, 'Mtt_std' : Mtt_std, 'Mpp' : Mpp, 'Mpp_std' : Mpp_std, 'Mrt' : Mrt, 'Mrt_std' : Mrt_std, 'Mrp' : Mrp, 'Mrp_std' : Mrp_std, 'Mtp' : Mtp, 'Mtp_std' : Mtp_std}

df = pd.DataFrame(data=d)

print(df)

df.sort_values('CMT_name', inplace = True)
df.drop_duplicates(subset = 'CMT_name', keep = False, inplace = True)

print(df)

df.to_pickle("./gcmt_all_earthquakes.pkl")
