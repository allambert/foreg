DEFAULT_TORCH_DTYPE = "float64"

N_SAMPLES = 100
N_THETA = 100
N_TEST = 100

SEED_GPS = 65
SEED = 7786

STDS_GPS_IN = [0.05, 0.1, 0.5, 0.7]
STDS_GPS_OUT = [0.05, 0.1, 0.5, 0.7]
GP_SCALE = 1.5

KERNEL_INPUT_GAMMA = 0.01
KERNEL_OUTPUT_GAMMA = 100.

LBDA_SEARCH = (-6, -3, 10)
CV_SPLIT = 5

CORRUPT_GLOBAL_PARAMS = {"freq_sample":0.1, "intensity": (0., 4., 10), "seed_gps": 56}
CORRUPT_GLOBAL_FREQ_PARAMS = {"freq_sample":(0., 0.2, 10), "intensity": 3., "seed_gps": 56}

CORRUPT_LOCAL_FREQ_PARAMS = {"freq_sample":(0, 1., 10), "intensity": 1., "freq_loc": 0.1}
CORRUPT_LOCAL_LOC_PARAMS = {"freq_sample":1., "intensity": 1., "freq_loc": (0, 0.3, 10)}


