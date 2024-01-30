import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import argparse
import numpy as np
from pathlib import Path
import json


# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se


data_folder = Path("../data")
results_folder = Path("../results")
scratch_folder = Path("../scratch")


# Define argument parser
parser = argparse.ArgumentParser(description="Dispatch jobs for AIND ephys pipeline")

concat_group = parser.add_mutually_exclusive_group()
concat_help = "Whether to concatenate recordings (segments) or not. Default: False"
concat_group.add_argument("--concatenate", action="store_true", help=concat_help)
concat_group.add_argument("static_concatenate", nargs="?", default="false", help=concat_help)


if __name__ == "__main__":
    args = parser.parse_args()

    CONCAT = args.concatenate or args.static_concatenate.lower() == "true"

    print(f"Running job dispatcher with the following parameters:")
    print(f"\tCONCATENATE RECORDINGS: {CONCAT}")

    # find ecephys sessions to process
    # for pipelines, the session data should to be mapped to the "data/ecephys_session" folder
    if (
        (data_folder / "ecephys").is_dir()
        or (data_folder / "ecephys_compressed").is_dir()
        or (data_folder / "ecephys_clipped").is_dir()
    ):
        ecephys_sessions = [data_folder]
    else:
        ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    print(f"Ecephys folders: {[str(s) for s in ecephys_sessions]}")

    # not needed, we can parallelize
    # assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    experiments_dict_list = []
    for session in ecephys_sessions:
        session_name = None
        if (session / "data_description.json").is_file():
            data_description = json.load(open(session / "data_description.json", "r"))
            session_name = data_description["name"]
        session_folder_path = session.relative_to(data_folder)

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

        print(
            f"Session: {session_name}\n\tSession path from data: {str(session_folder_path)} - Open Ephys folder: {str(ecephys_folder)}"
        )
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
        print("\tRecording to be processed in parallel:")
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
                            session_name=session_name,
                            session_folder_path=str(session_folder_path),
                        )
                        if CONCAT:
                            recording_name = f"{exp_stream_name}_recording"
                            job_dict["segment_index"] = None
                        else:
                            recording_name = f"{exp_stream_name}_recording{i_r + 1}"
                            job_dict["segment_index"] = i_r
                        total_duration = np.round(recording.get_total_duration(), 2)
                        print(f"\t\t{recording_name} - Duration: {total_duration} s")
                        job_dict["recording_name"] = recording_name
                        experiments_dict_list.append(job_dict)

    for i, experiment_dict in enumerate(experiments_dict_list):
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(experiment_dict, f, indent=4)
    print(f"Generated {len(experiments_dict_list)} job config files")
