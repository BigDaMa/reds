import shutil
import sys
sys.path.append("..")
from holoclean.holoclean import HoloClean, Session
from holoclean.errordetection.sql_dcerrordetector import SqlDCErrorDetection
from holoclean.errordetection.sql_nullerrordetector import SqlnullErrorDetection


# data_path = "data/hospital.csv"
# dc_path = "data/hospital_constraints.txt"
# gt_path = "data/hospital_clean.csv"

# data_path = "data/hosp_holoclean.csv"
# dc_path = "data/hosp_holoclean_constraints.txt"
# gt_path = "data/hospital_clean.csv"

# data_path = "data/flight_holoclean.csv"
# dc_path = "data/flight_constraints.txt"
# gt_path = "data/hospital_clean.csv"

# data_path = "data/address_10_capitalized.csv"
# dc_path = "data/address_ten_constraints.txt"
# gt_path = "data/address_10_ground_truth_capitalized.csv"

# data_path = "data/dirty.csv"
# dc_path = "data/movies_1_constraints.txt"
# gt_path = "data/hospital_clean.csv"

data_path = "data/dirty-beers-and-breweries.csv"
dc_path = "data/bears_constraints.txt"
gt_path = "data/address_10_ground_truth_capitalized.csv"

index_attribute = "index"

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
detector = SqlDCErrorDetection(session)
error_detector_list =[]
error_detector_list.append(detector)
clean, dirty = session.detect_errors(error_detector_list)
#clean.head(5)
#dirty.head(5)
repaired = session.repair()
repaired = repaired.withColumn(index_attribute, repaired[index_attribute].cast("int"))
repaired.sort(index_attribute)
shutil.rmtree("repaired")
# repaired.repartition(1).write.format('com.databricks.spark.csv').option("header", 'true').save('repaired')
repaired.coalesce(1).write.format('com.databricks.spark.csv').option("header", 'true').save('repaired')
# session.compare_to_truth(gt_path)
