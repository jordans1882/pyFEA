from optimizationproblems.field import Field
import pandas as pd
import pickle, os, re

"""
Create a field object to save.
Generates (or loads) the grid and calculates information for each of the grid cells.
"""

current_working_dir = os.getcwd()
path_ = re.search(r"^(.*?[\\/]FEA)", current_working_dir)
path = path_.group()

field_ = Field()
fieldname = "sec35middle"
field_.cell_width = 122
field_.cell_length_max = 350
field_.cell_length_min = 350
field_.field_shape_file = (
    "C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\sec35mid\\sec35mid_bbox.shp"
)
yld_file = pd.read_csv(
    "C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\broyles_10m_yldDat_with_sentinel.csv"
)
field_.yld_file = yld_file.loc[(yld_file["field"] == fieldname.lower())]
# field_.as_applied_file = "C:\\Users\\f24n127\\Documents\\Work\\Ag\\Data\\henrys\\wood_henrys_2018_AA_N.shp"
field_.create_field()
pickle.dump(field_, open(path + "/utilities/saved_fields/sec35middle.pickle", "wb"))
# field_1 = Field()
# print('35MID')
# field_1.field_shape_file = "/home/amy/Documents/Work/OFPE/Data/Sec35Mid/sec35mid_bbox.shp"
# field_1.yld_file = "/home/amy/Documents/Work/OFPE/Data/Sec35Mid/Broyles Farm_Broyles Fami_sec 35 middl_Harvest_2020-08-17_00.shp"
# field_1.as_applied_file = "/home/amy/Documents/Work/OFPE/Data/Sec35Mid/sec35mid_AA_N_2020_pts.shp"
# field_1.create_field()
# pickle.dump(field_1, open('/home/amy/projects/FEA/utilities/saved_fields/sec35mid.pickle', 'wb'))
# field_2 = Field()
# print('35WEST')
# field_2.field_shape_file = "../../../Documents/Work/OFPE/Data/Sec35West/sec35west_bbox.shp"
# field_2.yld_file = "../../../Documents/Work/OFPE/Data/Sec35West/Broyles Farm_Broyles Fami_sec 35 west_Harvest_2020-08-07_00.shp"
# field_2.as_applied_file = "../../../Documents/Work/OFPE/Data/Sec35West/sec35west_AA_N_2020_pts.shp"
# field_2.create_field()
# pickle.dump(field_2, open('../../utilities/saved_fields/sec35west.pickle', 'wb'))

# field_names = ['millview']
# for fieldname in field_names:
#     field_organic = Field()
#     field_organic.cell_length_min = 350
#     field_organic.cell_length_max = 600
#     field_organic.cell_width = 43
#     field_organic.angle = 90
#     field_organic.field_shape_file = "/home/amy/Documents/Work/OFPE/Data/organic/" + fieldname + "/bbox_mv_2020.shp"
#     df = pd.read_csv("/home/amy/Documents/Work/OFPE/Data/organic/all_agg_exp_2.csv")
#     field_organic.yld_file = df.loc[(df['field']==fieldname)]
#     field_organic.create_field()
#     pickle.dump(field_organic, open(path+'/utilities/saved_fields/' + fieldname + '.pickle', 'wb'))
