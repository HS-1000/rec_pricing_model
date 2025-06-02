import features

rec_path = "./data/rec.csv"
dem_path = "./data/demand.csv"

x, y = features.get_xy(rec_path=rec_path, demand_path=dem_path)
print(x.shape)
print(y.shape)