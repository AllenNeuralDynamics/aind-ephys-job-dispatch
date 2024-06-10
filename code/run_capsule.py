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

from spikeinterface.core.core_tools import SIJsonEncoder


data_folder = Path("../data")
results_folder = Path("../results")

# Define argument parser
parser = argparse.ArgumentParser(description="Dispatch jobs for AIND ephys pipeline")

concat_group = parser.add_mutually_exclusive_group()
concat_help = "Whether to concatenate recordings (segments) or not. Default: False"
concat_group.add_argument("--concatenate", action="store_true", help=concat_help)
concat_group.add_argument("static_concatenate", nargs="?", default="false", help=concat_help)

input_group = parser.add_mutually_exclusive_group()
input_help = "Which 'loader' to use (aind | spikeglx | nwb)"
input_group.add_argument("--input", default="aind", help=input_help, choices=["aind", "spikeglx", "nwb"])
input_group.add_argument("static_input", nargs="?", default="aind", help=input_help)


if __name__ == "__main__":
    args = parser.parse_args()

    CONCAT = True if args.static_concatenate and args.static_concatenate.lower() == "true" else args.concatenate
    INPUT = args.input or args.static_input

    print(f"Running job dispatcher with the following parameters:")
    print(f"\tCONCATENATE RECORDINGS: {CONCAT}")
    print(f"\tINPUT: {INPUT}")

    print(f"Parsing {INPUT} input folder")
    recording_dict = {}
    if INPUT == "aind":
        # find ecephys sessions to process
        #
        # - for pipelines, the session data should to be mapped to the "data/ecephys_session" folder
        # - for standalone capsule runs, the data is in "data/ecephys_{session_name}"
        ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower() or "behavior" in p.name.lower()]

        for session_folder in ecephys_sessions:
            session_name = None
            if (session_folder / "data_description.json").is_file():
                data_description = json.load(open(session_folder / "data_description.json", "r"))
                session_name = data_description["name"]

            # in the AIND pipeline, the session folder is mapped to 
            ecephys_base_folder = session_folder / "ecephys"

            if (ecephys_base_folder / "ecephys_compressed").is_dir():
                new_format = True
                ecephys_folder = ecephys_base_folder
            else:
                new_format = False
                ecephys_folder = session_folder

            compressed = False
            if (ecephys_folder / "ecephys_compressed").is_dir():
                # most recent folder organization
                compressed = True
                ecephys_compressed_folder = ecephys_folder / "ecephys_compressed"
                ecephys_openephys_folder = ecephys_folder / "ecephys_clipped"
            else:
                # uncompressed data
                ecephys_openephys_folder = ecephys_base_folder

            print(f"\tSession name: {session_name}")
            print(f"\tOpen Ephys folder: {str(ecephys_openephys_folder)}")
            if compressed:
                print(f"\tZarr compressed folder: {str(ecephys_compressed_folder)}")

            # get blocks/experiments and streams info
            num_blocks = se.get_neo_num_blocks("openephys", ecephys_openephys_folder)
            stream_names, stream_ids = se.get_neo_streams("openephys", ecephys_openephys_folder)

            # load first stream to map block_indices to experiment_names
            rec_test = se.read_openephys(ecephys_openephys_folder, block_index=0, stream_name=stream_names[0])
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
                            recording = se.read_openephys(ecephys_openephys_folder, stream_name=stream_name, block_index=block_index)
                        else:
                            recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")
                        recording_name = f"{exp_stream_name}_recording"
                        recording_dict[(session_name, recording_name)] = recording

    elif INPUT == "spikeglx":
        # get blocks/experiments and streams info
        spikeglx_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        assert len(spikeglx_folders) == 1, "Attach one SpikeGLX folder at a time"
        spikeglx_folder = spikeglx_folders[0]
        session_name = spikeglx_folder.name
        stream_names, stream_ids = se.get_neo_streams("spikeglx", spikeglx_folder)

        # spikeglx has only one block
        num_blocks = 1
        block_index = 0

        print(f"\tSession name: {session_name}")
        print(f"\tNum. streams: {len(stream_names)}")
        for stream_name in stream_names:
            if "nidq" not in stream_name and "lf" not in stream_name:
                recording = se.read_spikeglx(spikeglx_folder, stream_name=stream_name)
                recording_name = f"block{block_index}_{stream_name}_recording"
                recording_dict[(session_name, recording_name)] = recording

    elif INPUT == "nwb":
        # get blocks/experiments and streams info
        nwb_files = [p for p in data_folder.iterdir() if "nwb" in p.name]
        assert len(nwb_files) == 1, "Attach one NWB file at a time"
        nwb_file = nwb_files[0]
        session_name = nwb_file.name

        # spikeglx has only one block
        num_blocks = 1
        block_index = 0

        # get available electrical_series_path options
        # TODO: use NWBRecordingExtractor.fetch_available_electrical_series_paths with spikeinterface==0.101.0
        from spikeinterface.extractors.nwbextractors import _get_backend_from_local_file, _find_neurodata_type_from_backend, read_file_from_backend

        backend = _get_backend_from_local_file(nwb_file)
        file_handle = read_file_from_backend(
            file_path=nwb_file,
        )
        electrical_series_paths = _find_neurodata_type_from_backend(
            file_handle,
            neurodata_type="ElectricalSeries",
            backend=backend,
        )

        print(f"\tSession name: {session_name}")
        print(f"\tNum. Blocks {num_blocks} - Num. streams: {len(electrical_series_paths)}")
        for electrical_series_path in electrical_series_paths:
            # only use paths in acquisition
            if "acquisition" in electrical_series_path:
                stream_name = electrical_series_path.replace("/", "-")
                recording = se.read_nwb_recording(nwb_file, electrical_series_path=electrical_series_path)
                recording_name = f"block{block_index}_{stream_name}_recording"
                recording_dict[(session_name, recording_name)] = recording

    # populate job dict list
    job_dict_list = []
    print("Recording to be processed in parallel:")
    for session_recording_name in recording_dict:
        session_name, recording_name = session_recording_name
        recording = recording_dict[session_recording_name]

        if CONCAT:
            recordings = [recording]
        else:
            recordings = si.split_recording(recording)

        for segment_index, recording in enumerate(recordings):
            if not CONCAT:
                recording_name = f"{recording_name}{segment_index + 1}"
            duration = np.round(recording.get_total_duration(), 2)

            # if multiple channel groups, process in parallel
            if len(np.unique(recording.get_channel_groups())) > 1:
                for group_name, recording_group in recording.split_by("group").items():
                    recording_name_group = f"{recording_name}_group{group_name}"
                    print(f"\t{recording_name_group} - Duration: {duration} s - Num. channels: {recording_group.get_num_channels()}")
                    job_dict = dict(
                        session_name=session_name,
                        recording_name=str(recording_name_group),
                        recording_dict=recording_group.to_dict(
                            recursive=True,
                            relative_to=data_folder
                        )
                    )
                    job_dict_list.append(job_dict)
            else:
                print(f"\t{recording_name} - Duration: {duration} s - Num. channels: {recording.get_num_channels()}")
                job_dict = dict(
                    session_name=session_name,
                    recording_name=str(recording_name),
                    recording_dict=recording.to_dict(
                        recursive=True,
                        relative_to=data_folder
                    )
                )
            job_dict_list.append(job_dict)

    if not results_folder.is_dir():
        results_folder.mkdir(parents=True)

    for i, job_dict in enumerate(job_dict_list):
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)
    print(f"Generated {len(job_dict_list)} job config files")
