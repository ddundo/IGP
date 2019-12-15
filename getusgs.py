from datetime import datetime
from libcomcat.dataframes import get_detail_data_frame
from libcomcat.search import search, get_event_by_id


summary_events = search(starttime=datetime(2010, 1, 1, 00, 00), endtime=datetime(2019, 12, 31, 23, 59),
                        minlatitude=45, maxlatitude=72, minlongitude=-180, maxlongitude=-125, minmagnitude=4)
detail_df = get_detail_data_frame(summary_events, get_tensors='all', get_focals='all', get_moment_supplement=True)
detail_df.to_pickle('usgs_alaska_2010.p')

