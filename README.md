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
- `spikeinterface`: a [spikeinterface-supported format](https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html#raw-data-formats) (provide parsing info with the ``spikeinterface-info`` argument)
- `aind`: data folder created at AIND (e.g., "ecephys_664438_2023-04-12_14-59-51") with the `ecephys` folder containing
  the `ecepys_compressed` and `ecephys_clipped` folders. 

The ecephys data can have multiple experiments (or blocks), each with multiple streams (probes) and continuous recordings (segments). Different recordings can be concatenated together to form a single recording (and spike sorted together) or treated as separate recordings (and spike sorted separately).

### Parameters

The `code/run` script takes several arguments:

```bash
  --concatenate         Whether to concatenate recordings (segments) or not. Default: False
  --no-split-groups     Whether to process different groups separately
  --debug               Whether to run in DEBUG mode
  --debug-duration DEBUG_DURATION
                        Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled
  --skip-timestamps-check
                        Skip timestamps check
  --multi-session       Whether the data folder includes multiple sessions or not. Default: False
  --input {aind,spikeglx,openephys,nwb,spikeinterface}
                        Which 'loader' to use (aind | spikeglx | openephys | nwb | spikeinterface)
  --spikeinterface-info SPIKEINTERFACE_INFO
                        A JSON path or string to specify how to parse the recording in spikeinterface, including: 
                        - 1. reader_type (required): string with the reader type (e.g. 'plexon', 'neuralynx', 'intan' etc.).
                        - 2. reader_kwargs (optional): dictionary with the reader kwargs (e.g. {'folder': '/path/to/folder'}).
                        - 3. keep_stream_substrings (optional): string or list of strings with the stream names to load (e.g. 'AP' or ['AP', 'LFP']).
                        - 4. skip_stream_substrings (optional): string (or list of strings) with substrings used to skip streams (e.g. 'NIDQ' or ['USB', 'EVENTS']).
                        - 5. probe_paths (optional): string or dict the probe paths to a ProbeInterface JSON file (e.g. '/path/to/probe.json'). If a dict is provided, the key is the stream name and the value is the probe path. If reader_kwargs is not provided, the reader will be created with default parameters. The probe_path is required if the reader doesn't load the probe automatically.
```

### Output

The output of this capsule is a list of JSON files in the `results/` folder, containing the parameters for a spike sorting job. 

Each JSON file contains the following fields:

- `session_name`: the session name (e.g., "ecephys_664438_2023-04-12_14-59-51")
- `recording_name`: the recording name, which will correspond to output folders downstreams (e.g, "experiment1_Record Node 101#Neuropix-PXI-100.probeA-AP_recording1")
- `recording_dict`: the SpikeInterface dict representation of the recording with paths relative to the `data` folder
- `recording_lfp_dict`: the SpikeInterface dict representation of the LFP recording associated to the main recording (e.g., the LF stream in case of Neuropixels). If the recording is wideband this field will be set.