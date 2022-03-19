from sys import platform

# data processing configs

if "linux" in platform.lower():
    raw_data_folder = "/home/michael/SSD_Cache/PEAR/organized_raw"
    processed_data_folder = "/home/michael/SSD_Cache/PEAR/processed_data"
    log_path = "/home/michael/SSD_Cache/PEAR/train_logs"
else:
    raw_data_folder = "Y:/PEAR/organized_raw"
    processed_data_folder = "Y:/PEAR/processed_data"
    log_path = "Y:/PEAR/train_logs"

# reporting config
line_width = 85
