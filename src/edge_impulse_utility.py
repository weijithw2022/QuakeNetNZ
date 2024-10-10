## Utilities to explore functionalities with edge impulse

from utils import *
from config import *

# This function converts hdf5 file data to CSV that can use for edge impulse data
def convertHDF5ToCSV(cfg):
    # Open the HDF5 file
    os.makedirs(cfg.EDGE_IMPULSE_CSV_PATH, exist_ok=True)

    # Create negative and positive directories
    positive_dir = cfg.EDGE_IMPULSE_CSV_PATH +"/positive"
    negative_dir = cfg.EDGE_IMPULSE_CSV_PATH +"/negative"
    
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)

    with h5py.File(cfg.DATABASE_FILE, 'r') as hdf:
        groups = ["positive_samples_p", "positive_samples_s", "negative_sample_group"]
        
        for group in groups:
            group_data = hdf[group]
            
            for event_id in group_data.keys():
                # Extract the dataset for each event
                dataset = group_data.get(event_id)
                data = np.array(dataset)  # Shape (3, timesteps)
                sampling_rate = dataset.attrs["sampling_rate"]

                # Create a timestamp column
                num_samples = data.shape[1]
                time_step = 1 / sampling_rate
                timestamps = np.arange(0, num_samples * time_step, time_step)[:num_samples]
                    
                # Convert to DataFrame with timestamps
                df = pd.DataFrame(data.T, columns=["X", "Y", "Z"])
                df.insert(0, "timestamp", timestamps)  # Insert timestamp column at the start
                
                # Save each event as a CSV file
                if group == "positive_samples_p" or group ==  "positive_samples_s":
                    output_file = os.path.join(positive_dir, f"{group}_{event_id}.csv")
                else:
                    output_file = os.path.join(negative_dir, f"{group}_{event_id}.csv")
                
                df.to_csv(output_file, index=False)
                print(f"Saved: {output_file}")

def main():
    cfg = Config()
    convertHDF5ToCSV(cfg)

if __name__ == "__main__":
    main()
