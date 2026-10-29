# Dispatch jobs for AIND ephys pipeline
## aind-ephys-job-dispatch


### Description

This simple capsule is designed to dispatch jobs for the AIND pipeline. 

It assumes the data is stored in the `data/` directory, and creates as many JSON files 
as the number of jobs that can be run in parallel. Each job consists of a recording with spiking activity that needs spike sorting.

### Inputs

The `data/` folder must include an `ecephys` folder with a single recorded session.

The data ingestion layer supports multiple input formats:

- `spikeglx`: SpikeGLX generated folder
- `openephys`: OpenEphys generated folder (Neuropixels plugin required)
- `nwb`: NWB file (either in HDF5 or Zarr backend)
- `spikeinterface`: a [spikeinterface-supported format](https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html#raw-data-formats) (provide parsing info with the ``spikeinterface_info`` parameter)
- `aind`: data folder created at AIND (e.g., "ecephys_664438_2023-04-12_14-59-51") with the `ecephys` folder containing
  the `ecepys_compressed` and `ecephys_clipped` folders. 

The ecephys data can have multiple experiments (or blocks), each with multiple streams (probes) and continuous recordings (segments). Different recordings can be concatenated together to form a single recording (and spike sorted together) or treated as separate recordings (and spike sorted separately).

### Parameters

Parameters are read from a `params.json` file that must be in the root folder of the capsule (next to `code/`, i.e. `../params.json` from `code/`), with the step parameters in the `job_dispatch` section and the optional `logging` section:

```json
{
    "logging": {"package": "logging"},
    "job_dispatch": {
        "input": "aind",
        "split_segments": false,
        "split_groups": true,
        "debug": false,
        "debug_duration": 30,
        "skip_timestamps_check": false,
        "multi_session": false,
        "min_recording_duration": -1
    }
}
```

- `input` (required): which 'loader' to use (`aind` | `spikeglx` | `openephys` | `nwb` | `spikeinterface`)
- `split_segments`: whether to split (true) or concatenate (false) recording segments. Default: false
- `split_groups`: whether to process different groups separately. Default: true
- `debug`: whether to run in DEBUG mode. Default: false
- `debug_duration`: duration (s) of the clipped recording in debug mode. Only used if `debug` is true
- `skip_timestamps_check`: skip the timestamps check. Default: false
- `multi_session`: whether the data folder includes multiple sessions. Default: false
- `min_recording_duration`: recordings shorter than this (s) are skipped. Default: -1 (no minimum)
- `nwb_files`: comma-separated paths to NWB files (only for `input="nwb"`). If not set, they are inferred from the data folder
- `spikeinterface_info` (required for `input="spikeinterface"`): a JSON path or string specifying how to parse the recording in spikeinterface, including:
  - `reader_type` (required): string with the reader type (e.g. 'plexon', 'neuralynx', 'intan' etc.).
  - `reader_kwargs` (optional): dictionary with the reader kwargs (e.g. {'folder': '/path/to/folder'}).
  - `keep_stream_substrings` (optional): string or list of strings with the stream names to load (e.g. 'AP' or ['AP', 'LFP']).
  - `skip_stream_substrings` (optional): string (or list of strings) with substrings used to skip streams (e.g. 'NIDQ' or ['USB', 'EVENTS']).
  - `probe_paths` (optional): string or dict with the probe paths to a ProbeInterface JSON file (e.g. '/path/to/probe.json'). If a dict is provided, the key is the stream name and the value is the probe path. The probe path is required if the reader doesn't load the probe automatically.

### Output

The output of this capsule is a list of JSON files in the `results/` folder, containing the parameters for a spike sorting job. 

Each JSON file contains the following fields:

- `session_name`: the session name (e.g., "ecephys_664438_2023-04-12_14-59-51")
- `recording_name`: the recording name, which will correspond to output folders downstreams (e.g, "experiment1_Record Node 101#Neuropix-PXI-100.probeA-AP_recording1")
- `recording_dict`: the SpikeInterface dict representation of the recording with paths relative to the `data` folder
- `recording_lfp_dict`: the SpikeInterface dict representation of the LFP recording associated to the main recording (e.g., the LF stream in case of Neuropixels). If the recording is wideband this field will be set.