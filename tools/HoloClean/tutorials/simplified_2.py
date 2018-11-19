import sys
import shutil
import numpy
import pandas
sys.path.append("..")
from holoclean.holoclean import HoloClean, Session


class SimpleErrorDetector:
    def __init__(self, session):
        self.spark_session = session.holo_env.spark_session
        self.dataengine = session.holo_env.dataengine
        self.dataset = session.dataset
    
    def get_noisy_cells(self):
        ddf = pandas.read_csv(data_path, sep=",", header="infer", encoding="utf-8", dtype=str, keep_default_na=False, low_memory=False, index_col=index_attribute)
        ddf = ddf.apply(lambda x: x.str.strip())
        gdf = pandas.read_csv(gt_path, sep=",", header="infer", encoding="utf-8", dtype=str, keep_default_na=False, low_memory=False, index_col=index_attribute)
        gdf = gdf.apply(lambda x: x.str.strip())
        gdf.columns = ddf.columns 
        ne_stacked = (ddf != gdf).stack()
        changed = ne_stacked[ne_stacked]
        result = session.holo_env.spark_session.createDataFrame(changed.keys(), ["ind", "attr"])
        return result      
        
    def get_clean_cells(self):
        ddf = pandas.read_csv(data_path, sep=",", header="infer", encoding="utf-8", dtype=str, keep_default_na=False, low_memory=False, index_col=index_attribute)
        ddf = ddf.apply(lambda x: x.str.strip())
        gdf = pandas.read_csv(gt_path, sep=",", header="infer", encoding="utf-8", dtype=str, keep_default_na=False, low_memory=False, index_col=index_attribute)
        gdf = gdf.apply(lambda x: x.str.strip())
        gdf.columns = ddf.columns 
        ne_stacked = (ddf == gdf).stack()
        changed = ne_stacked[ne_stacked]
        result = session.holo_env.spark_session.createDataFrame(changed.keys(), ["ind", "attr"])
        return result


index_attribute = "index"

data_path = "../../../../red/datasets/hosp_holoclean_new/hospital.csv"
gt_path = "../../../../red/datasets/hosp_holoclean_new/clean.csv"
dc_path = "../../../../red/datasets/hosp_holoclean_new/holoclean_constraints.txt"

# data_path = "../../red/datasets/flight_holoclean/flight_holoclean.csv"
# gt_path = "../../red/datasets/flight_holoclean/flight.csv"
# dc_path = "../../red/datasets/flight_holoclean/holoclean_constraints.txt"

# data_path = "../../red/datasets/address_10_capitalized/address_10_capitalized.csv"
# gt_path = "../../red/datasets/address_10_capitalized/address_10_ground_truth_capitalized.csv"
# dc_path = "../../red/datasets/address_10_capitalized/holoclean_constraints.txt"

# data_path = "../../red/datasets/bears/dirty-beers-and-breweries.csv"
# gt_path = "../../red/datasets/bears/beers-and-breweries.csv"
# dc_path = "../../red/datasets/bears/holoclean_constraints.txt"


holo = HoloClean(
	holoclean_path="..",   # path to holoclean package
    verbose=False,
    pruning_threshold1=0.1,   # to limit possible values for training data
    pruning_clean_breakoff=6,   # to limit possible values for training data to less than k values
    pruning_threshold2=0,   # to limit possible values for dirty data (applied after Threshold 1)
    pruning_dk_breakoff=6,   # to limit possible values for dirty data to less than k values
    learning_iterations=30,   # learning parameters
    learning_rate=0.001,
    batch_size=5)
session = Session(holo)
data = session.load_data(data_path)
dcs = session.load_denial_constraints(dc_path)
#data.select('City').show(15)
detector = SimpleErrorDetector(session)
error_detector_list =[]
error_detector_list.append(detector)
clean, dirty = session.detect_errors(error_detector_list)
# clean.head(5)
# dirty.head(5)
repaired = session.repair()
repaired = repaired.withColumn(index_attribute, repaired[index_attribute].cast("int"))
repaired.sort(index_attribute)
shutil.rmtree("repaired")
# repaired.repartition(1).write.format('com.databricks.spark.csv').option("header", 'true').save('repaired')
repaired.coalesce(1).write.format('com.databricks.spark.csv').option("header", 'true').save('repaired')
# session.compare_to_truth(gt_path)
