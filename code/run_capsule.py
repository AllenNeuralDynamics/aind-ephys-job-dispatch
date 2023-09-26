import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import os
import numpy as np
from pathlib import Path
import shutil
import json
import sys
import time
from datetime import datetime, timedelta


# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se


data_folder = Path("../data")
results_folder = Path("../results")
scratch_folder = Path("../scratch")


if __name__ == "__main__":

    if len(sys.argv) == 2:
        if sys.argv[1] == "true":
            CONCAT = True
        else:
            CONCAT = False
    else:
        CONCAT = False

    # find ecephys sessions to process
    # for pipelines, the session data are mapped directly to the data folder
    if (data_folder / "ecephys").is_dir() or (data_folder / "ecephys_compressed").is_dir():
        ecephys_sessions = [data_folder]
    else:
        ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    print(f"Ecephys sessions: {ecephys_sessions}")

    # not needed, we can parallelize
    # assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    experiments_dict_list = []
    for session in ecephys_sessions:
        session_name = session.name
        # in pipeline mode, we can't retrieve the session name from the folder name.
        if session_name == "data":
            session_name = None

        ecephys_base_folder = session / "ecephys"
        compressed = False
        if (ecephys_base_folder / "ecephys_compressed").is_dir():
            # most recent folder organization
            compressed = True
            ecephys_compressed_folder = ecephys_base_folder / "ecephys_compressed"
            ecephys_folder = ecephys_base_folder / "ecephys_clipped"
        elif (session / "ecephys_compressed").is_dir():
            compressed = True
            ecephys_compressed_folder = session / "ecephys_compressed"
            ecephys_folder = session / "ecephys_clipped"
        else:
            # uncompressed data
            ecephys_folder = ecephys_base_folder

        print(f"Session: {session_name} - Open Ephys folder: {ecephys_folder}")
        # get blocks/experiments and streams info
        num_blocks = se.get_neo_num_blocks("openephys", ecephys_folder)
        stream_names, stream_ids = se.get_neo_streams("openephys", ecephys_folder)

        # load first stream to map block_indices to experiment_names
        rec_test = se.read_openephys(ecephys_folder, block_index=0, stream_name=stream_names[0])
        record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
        experiments = rec_test.neo_reader.folder_structure[record_node]["experiments"]
        exp_ids = list(experiments.keys())
        experiment_names = [experiments[exp_id]["name"] for exp_id in sorted(exp_ids)]

        print(f"\tNum. Blocks {num_blocks} - Num. streams: {len(stream_names)}")

        for block_index in range(num_blocks):
            for stream_name in stream_names:
                # skip NIDAQ and NP1-LFP streams
                if "NI-DAQ" not in stream_name and "LFP" not in stream_name and "Rhythm" not in stream_name:
                    experiment_name = experiment_names[block_index]
                    exp_stream_name = f"{experiment_name}_{stream_name}"

                    if not compressed:
                        recording = se.read_openephys(ecephys_folder, stream_name=stream_name, block_index=block_index)
                    else:
                        recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")

                    if CONCAT:
                        recordings = [recording]
                    else:
                        recordings = si.split_recording(recording)

                    for i_r, recording in enumerate(recordings):
                        job_dict = dict(
                            experiment_name=experiment_name,
                            block_index=block_index,
                            stream_name=stream_name,
                            session=session_name
                        )
                        if CONCAT:
                            recording_name = f"{exp_stream_name}_recording"
                            job_dict["segment_index"] = None
                        else:
                            recording_name = f"{exp_stream_name}_recording{i_r + 1}"
                            job_dict["segment_index"] = i_r
                        print(f"\t\t{recording_name}")
                        job_dict["recording_name"] = recording_name
                        experiments_dict_list.append(job_dict)

    for i, experiment_dict in enumerate(experiments_dict_list):
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(experiment_dict, f, indent=4)
    print(f"Generated {len(experiments_dict_list)} job config files")

