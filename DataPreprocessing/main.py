import utils

edf_dir = "DATADIRECTORY"      # Replace with your actual data directory path
model_path = "MODELDIRECTORY"  # Replace with the trained model's path

# Set the start and end timestamps for the segment you want to preprocess
start = "segment start timestamp"
end = "segment end timestamp"

raw_data, preprocessed_data = utils.preprocess_data(
    edf_dir,
    model_path,
    start,
    end
)
