# CS224W: solar hosting-capacity dataset

An end-to-end pilot for **independent PV additions** at customer buses on one
SMART-DS SFO feeder. It downloads source files, selects a load snapshot, extracts
a rooted graph, calculates hosting-capacity labels with OpenDSS, and exports
PyTorch Geometric tensors. SweepGNN and an optional training entry point are
included; dataset curation does not start training.

## Run locally on macOS

The pilot was exercised on Apple Silicon with Python 3.10. OpenDSSDirect bundles
the DSS-Extensions engine; a separate Windows OpenDSS installation is unnecessary.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/solar-hc run
```

The environment is already installed in this checkout. The source download is
about **36 MB**, plus Python dependencies. Raw data and generated outputs are
ignored by Git. `requirements-macos-arm64.lock` records the tested environment;
to reproduce its dependency versions, install it before installing this package.

For a short first run or to resume labeling:

```sh
.venv/bin/solar-hc run --max-buses 2
.venv/bin/solar-hc label
```

`--max-buses` limits the number of **additional** buses processed in that command.
Each completed bus is checkpointed atomically. Repeating `run` verifies cached
source checksums and reuses the prepared snapshot and completed labels.

Individual stages are also available:

```sh
.venv/bin/solar-hc download
.venv/bin/solar-hc prepare
.venv/bin/solar-hc label
.venv/bin/solar-hc validate
.venv/bin/python -m pytest -q
```

Settings live in `configs/sfo_pilot.json`. Changing source, settings, pipeline code,
or dependency versions invalidates checkpoint reuse. Use a new `data_dir` in a
copied config for a new experiment, then pass `--config path/to/config.json`.
Relative data paths resolve from the config directory's parent.

## Pilot definition

- **Source:** SMART-DS v1.0 / 2016 / SFO / P1U / `base_timeseries`, feeder
  `p1uhs0_1247--p1udt942`. Source files remain unchanged. Only referenced DSS files
  and annual CSV profiles are downloaded from the public OEDI bucket.
- **Graph:** 84 buses, 83 physical components, maximum depth 17, and 26 customer
  buses eligible for labels. The root is the feeder-head voltage source, which
  substitutes for the upstream substation in this standalone feeder model.
- **Snapshot:** rank coincident 15-minute samples between 11:00 and 14:00 model
  time by total real demand; choose the lowest-load baseline-feasible sample.
  Rejected samples are retained in `baseline_audit.json`. Profiles use model time;
  no geographic timezone, weather guarantee, or leap-day correction is inferred.
- **PV:** this source scenario has **zero existing PV**. New PV delivers its
  tested AC kW at unity power factor. Capacity is AC injection, not DC panel
  nameplate. Each bus is tested independently with all other buses at baseline.
- **Connections:** injection follows the customer's source voltage, connection
  type and phase terminals. Total bus injection is divided equally among distinct
  load connections. A split-phase customer with two 120 V legs therefore receives
  half the tested kW on each leg. Repeated loads on the same connection do not
  multiply injected power.
- **Controls:** allow baseline controls to settle, then freeze their state during
  the PV search. Load P/Q profiles are evaluated at the same sample and frozen.
  New PV uses constant P and Q=0 within the voltage-feasible range.
- **Feasibility:** all represented non-ground bus voltages must be in
  `[0.95, 1.05]` p.u. All non-ground line conductors and transformer windings must
  stay within normal current ratings. Line limits must be explicitly declared in
  the source line or linecode. Transformer limits use `NormHkVA`, each winding's
  kVA/kV, and the normal-to-primary-nameplate ratio. No implicit default rating is
  accepted. Protection coordination and equipment aging are outside this target.
- **Search:** start at 5 kW, double until a physical violation or a 2,000 kW
  ceiling, audit 17 points across the discovered interval, then bisect to a bracket
  at most 0.5 kW wide. The target is the **feasible lower bound**. Feasibility is
  assumed contiguous from zero until its first violation; the sampled audit can
  detect some departures but does not prove monotonicity or examine later feasible
  islands. These are **snapshot** labels, not annual hosting-capacity guarantees.

The original master launches 35,040 time steps and plotting commands. The pipeline
creates a separate `snapshot_master.dss` that loads the model without executing
those commands.

## Outputs

Under `data/processed/SFO_P1U_p1uhs0_1247--p1udt942/`:

| File | Contents |
| --- | --- |
| `snapshot.json` | Coincident per-load P/Q, sample index, baseline checks, engine version, thermal limits |
| `baseline_audit.json` | Snapshot selection and any rejected baselines |
| `graph.json` | Bus/component metadata, winding connections, impedance matrices, rooted topology |
| `labels.json` | Resumable per-bus results and all sampled feasibility checks |
| `labels.csv` | Compact label table with search bounds, statuses and solve times |
| `graph.pt` | Tensor dictionary loadable with `weights_only=True` |
| `feature_schema.json` | Feature ordering, units, masks, phase and batching conventions |
| `splits.json` | Feeder-grouped split manifest |
| `run.json` | Configuration, source hashes, pipeline hashes, dependency versions, fingerprint |
| `summary.json` | Label counts, HC range, topology and simulation timings |
| `validation.json` | Fresh-engine re-solves of every saved valid label's feasible/infeasible bracket |

The download manifest and SHA-256 hashes are under `data/raw/<feeder_id>/`.

Only `status=ok` creates a regression label. Other outcomes include
`baseline_violation`, `solver_failure`, `missing_ratings`, `nonmonotonic`, and
`ceiling_reached`. A ceiling result supplies a lower bound but is excluded from
ordinary regression. Baseline selection failures stop preparation and write an
audit; incompatible topology stops graph extraction.

`label_mask` selects valid targets; unlabeled `y=0` values are placeholders.
`candidate_mask` selects customer buses. A genuine bracketed zero-capacity target
can still have `label_mask=True`.

```python
import torch
from torch_geometric.data import Data

path = "data/processed/SFO_P1U_p1uhs0_1247--p1udt942/graph.pt"
data = Data.from_dict(torch.load(path, weights_only=True))
print(data.x.shape, data.edge_index.shape, data.label_mask.sum())
# Later: loss = criterion(prediction[data.label_mask], data.y[data.label_mask])
```

Edges are bidirectional for standard GNNs. `edge_direction`, `parent_index`,
`depth`, `up_order`, and `down_order` also support tree sweeps. Physical matrix
terminal ordering and local phase mappings are retained. These are raw features;
fit scaling on training feeders only. Tree traversal indices are local to this
graph; a future multi-feeder loader must offset them explicitly when batching.

One feeder is assigned to **train only**, with empty validation/test sets.
Snapshots and scenarios of a physical feeder must remain together when expanding
to the planned 70/15/15 feeder split. This pilot cannot measure generalization.

The extractor checks for a connected radial multigraph and preserves split-phase
three-winding transformers as one edge when they connect two distinct buses.
Meshed/disconnected networks, open/partially open lines, unsupported series
components, and transformers with more than two distinct terminal buses cause
explicit errors rather than topology simplification. Existing PV is supported as
described below. Battery storage and non-PV generator scenarios are still rejected.

## Existing PV scenarios

The solar-low version of the same pilot feeder is configured separately:

```sh
.venv/bin/solar-hc run --config configs/sfo_pilot_pv_low.json
.venv/bin/solar-hc validate --config configs/sfo_pilot_pv_low.json
```

It writes to `data/solar_low/` and preserves the original zero-PV experiment.
The physical `feeder_id` remains identical across scenarios, so train/validation/
test splitting keeps those related graphs together.

Existing PV uses the **same profile sample as demand**. Its irradiance multiplier
and optional module-temperature profile are evaluated at that sample. Constant
weather is retained where no shape is specified. The native OpenDSS PV model
applies its source Pmpp, inverter kVA rating, efficiency/temperature curves, power
factor/kvar mode, cut-in/out, and enabled inverter controls during the baseline
solve. All time-varying profiles must share the normalized 15-minute time axis;
unsupported profile formats fail explicitly.

After baseline controls settle, each existing PV's actual terminal P/Q is recorded
and held fixed using a constant-P/Q source with the same electrical connection.
Baseline voltages must agree with the native PV solve to within `1e-5` p.u.
Transformer taps and capacitor states are saved for exact snapshot replay. The HC
search then varies **only the new candidate injection**. Existing inverter controls
do **not** readjust during that search; this is a fixed-dispatch snapshot target,
not hosting capacity with actively responding smart inverters.

The snapshot distinguishes **AC output at the selected time**, DC Pmpp nameplate,
and inverter kVA. For the solar-low pilot, the source has two PV systems with
13 kW DC nameplate, but output at a particular timestamp may be much lower.
Per-bus PV connections and P/Q are retained in `graph.json`; aggregate values and
source inverter-control definitions are retained in `snapshot.json`.

New exports use **15 node features**: the original 14 plus `existing_pv_kvar`.
`existing_pv_kw` now contains the actual existing AC injection. SweepGNN infers
input dimensions from the exported schema. Old 14-feature exports/checkpoints
remain readable, but cannot be mixed with new 15-feature exports in one run.
To regenerate the zero-PV pilot with this schema while retaining its old files:

```sh
.venv/bin/solar-hc run --data-dir data/base_v2
```

Dataset checkpoints reject changed source/code/settings. The new `--data-dir`
override lets each experiment use a separate directory without modifying a config.

To inspect existing PV and the new HC labels locally:

```sh
python3 scripts/visualize_feeder.py --data-dir data/solar_low/processed/SFO_P1U_p1uhs0_1247--p1udt942 --output outputs/visualizations/feeder-pv-low.html --open
```

Choose **Existing PV output** in the color selector, or hover over a bus to see
its existing kW/kvar output.

## Local graph visualization

Generate and open an interactive graph in your browser:

```sh
python3 scripts/visualize_feeder.py --open
```

The script uses only Python's standard library and reads the saved `graph.json`
and optional `labels.json`; it does not run simulations or training. The generated
`outputs/visualizations/feeder-local.html` works offline and can also be opened
directly or shared. The adjacent `scripts/feeder_viewer.html` is its reusable
template, so keep that file with the script.

Hover over a node for its type, nominal voltage, local demand, baseline voltage,
and HC when labeled. Switch between node roles, voltage levels, and HC; click a bus or use the dropdown
to highlight its path to the feeder head. A simplified customer connection shows
the transformer and voltage change. The page also explains the component types.

For another processed feeder or a different destination:

```sh
python3 scripts/visualize_feeder.py --data-dir data/processed/FEEDER_ID --output outputs/my-feeder.html --open
```

## SweepGNN

The implementation in `src/solar_hc/learning/model.py` follows the proposal:

1. **Upward:** each node combines its features with the **sum** of messages from
   its children, from leaves to root.
2. **Downward:** each node combines its upward state with a message from its
   parent's downward state, from root to leaves. A learned context vector replaces
   the missing parent at the root.
3. **Readout:** concatenate both states and predict nonnegative HC in AC kW.

Each direction has separate shared MLPs. Nodes at the same depth run together;
there is no fixed message-passing layer limit or recursive Python traversal.
The batch loader preserves the two component directions and handles multiple
trees, different depths and node orderings without cross-feeder messages.

To use the untrained model directly:

```python
from solar_hc.learning import SweepGNN
from solar_hc.learning.data import load_graph, collate_trees

graph = load_graph("data/processed/SFO_P1U_p1uhs0_1247--p1udt942/graph.pt")
batch = collate_trees([graph])
model = SweepGNN(node_dim=graph.x.shape[1], edge_dim=graph.down_attr.shape[1])
prediction_kw = model(batch)  # One prediction per bus, in the export's node order.
```

### Training when ready

No training starts on import. To launch it explicitly later:

```sh
.venv/bin/python -m solar_hc.learning.train --config configs/sweepgnn.json
```

The default config uses the existing pilot and a 64-dimensional hidden state.
Training uses AdamW, gradient clipping, and MAE **only on `label_mask=True`**.
Predictions and reported errors are in kW; the loss divides MAE by a fixed 100 kW
scale to control gradient magnitude. A softplus head guarantees nonnegative
predictions. Node/edge standardization is fitted using **training graphs only**
and saved as model buffers.

The loader reads explicit graph paths and a feeder split manifest. Multiple
snapshots of the same feeder follow that feeder's assignment. All loaded graphs
must use identical feature ordering, and feeders cannot appear in multiple splits.
Training paths resolve relative to the training config's own directory.

With the current single feeder, only training metrics are available; early
stopping and best-model selection stay disabled. Once validation feeders are
listed, validation MAE selects `best.pt` and controls early stopping. Test metrics
are computed only after training, using the selected checkpoint. The training
entry point never launches OpenDSS or regenerates data.

Outputs go to the configured experiment directory, initially `runs/sweepgnn/`:

- `last.pt`: final model, normalization, optimizer state, feature schema and config.
- `best.pt`: best validation checkpoint, produced only with validation feeders.
- `run.json`, `history.json`, `summary.json`: data hashes, split counts, settings,
  epoch metrics, timings and final evaluation.

The loop reports MAE/RMSE and false approval rates for 5/10/20 kW requests. A false
approval means predicted HC >= request while the conservative HC label < request;
the denominator is all approvals. A rate with no approvals is `null`. These rates
use the conservative point labels, not additional request-specific simulations.

Existing experiment outputs are protected from overwrite. For a new experiment,
use `--output-dir runs/another-experiment`. Saved optimizer state is available for
future resume support; the current CLI starts a fresh training run.

To load a trained checkpoint:

```python
import torch
from solar_hc.learning.model import load_checkpoint

model, metadata = load_checkpoint("runs/sweepgnn/last.pt")
with torch.no_grad():
    prediction_kw = model(batch)
```

## Sources

- [SMART-DS OEDI dataset](https://data.openei.org/submissions/2981)
- [SMART-DS user guide](https://oedi-data-lake.s3.amazonaws.com/SMART-DS/v1.0/User_Guide/Readme.pdf)
- [OpenDSSDirect installation and macOS support](https://dss-extensions.org/OpenDSSDirect.py/notebooks/Installation.html)
- [OpenDSS control modes](https://opendss.epri.com/ControlMode.html)
