import os
import random
import warnings
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import Aer
    from qiskit.converters import circuit_to_dag
    from qiskit.quantum_info import state_fidelity, Statevector, Operator
    from qiskit.circuit.library import XGate, HGate, SGate, TGate
    from qiskit.circuit import CircuitInstruction

    QISKIT_AVAILABLE = True
    BACKEND = Aer.get_backend("aer_simulator")

except ImportError:
    QISKIT_AVAILABLE = False
    print("[WARN] Qiskit not found.")


# =============================================================================
# CONFIG
# =============================================================================

ROOT_FOLDER = "data/circuits"
OUTPUT_DIR = "data"
PLOT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_QUBITS_STRUCT = 40
MAX_GATES = 2000
MAX_QUBITS_OIS = 14
MAX_QUBITS_FIDELITY = 12

SHOTS = 1024
SEVERITY_LEVELS = [0.1, 0.3, 0.6]
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# SIS WEIGHTS
# =============================================================================

WG, WD, WC, WT = 0.25, 0.25, 0.25, 0.25


# =============================================================================
# IGS-L WEIGHTS
# =============================================================================

W_EDGE = 0.15
W_NODE = 0.35
W_ORDER = 0.20
W_INTERACTION = 0.20
W_USAGE = 0.10

assert abs(
    W_EDGE + W_NODE + W_ORDER + W_INTERACTION + W_USAGE - 1.0
) < 1e-9, "IGS-L weights must sum to 1.0"


# =============================================================================
# LABELS
# =============================================================================

ANOMALY_TYPES = [
    "none",
    "gate_deletion_1q",
    "gate_deletion_2q",
    "gate_insertion",
    "gate_substitution",
    "gate_reorder",
    "trojan_NOT",
    "trojan_H",
    "qubit_swap",
]

ANOMALY_ORDER = [
    "gate_deletion_1q",
    "gate_deletion_2q",
    "gate_insertion",
    "gate_substitution",
    "gate_reorder",
    "trojan_NOT",
    "trojan_H",
    "qubit_swap",
]

ANOMALY_SIS_COMPONENT = {
    "gate_deletion_1q": "gate_count, depth",
    "gate_deletion_2q": "gate_count, CNOT_count, depth",
    "gate_insertion": "gate_count, depth increase",
    "gate_substitution": "DAG_topology only",
    "gate_reorder": "DAG_topology only",
    "trojan_NOT": "gate_count only",
    "trojan_H": "gate_count only",
    "qubit_swap": "CNOT_count, DAG_topology",
}

FAMILY_MAP = {
    "adder": "Arithmetic",
    "multiplier": "Arithmetic",
    "add": "Arithmetic",
    "qft": "Linear Algebra",
    "qpe": "Linear Algebra",
    "hhl": "Linear Algebra",
    "grover": "Oracle",
    "bv": "Oracle",
    "simon": "Oracle",
    "deutsch": "Oracle",
    "dj": "Oracle",
    "qaoa": "Variational",
    "vqe": "Variational",
    "bell": "State Prep",
    "ghz": "State Prep",
    "cat": "State Prep",
    "wstate": "State Prep",
    "w_state": "State Prep",
    "teleportation": "Communication",
    "bb84": "Communication",
    "ising": "Simulation",
    "trotter": "Simulation",
    "qec": "Error Correction",
    "lpn": "Error Correction",
    "shor": "Error Correction",
}

sns.set_theme(style="whitegrid", font_scale=1.0)


def save(fname, dpi=220):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {os.path.join(PLOT_DIR, fname)}")


def get_family(name):
    nl = name.lower()
    for key, fam in FAMILY_MAP.items():
        if key in nl:
            return fam
    return "Other"


# =============================================================================
# SIS
# =============================================================================

def _dag_feature_vector(qc):
    dag = circuit_to_dag(qc)
    nodes = list(dag.topological_op_nodes())

    indeg = np.array([len(list(dag.predecessors(n))) for n in nodes]) if nodes else np.array([0])
    outdeg = np.array([len(list(dag.successors(n))) for n in nodes]) if nodes else np.array([0])

    gc = len(qc.data)
    cx = qc.count_ops().get("cx", 0)

    return np.array([
        float(len(nodes)),
        float(gc),
        float(qc.depth()),
        float(cx),
        float(cx) / max(gc, 1),
        float(np.mean(indeg)),
        float(np.mean(outdeg)),
        float(np.max(indeg)),
        float(np.max(outdeg)),
    ])


def compute_sis(ref, test, wg=WG, wd=WD, wc=WC, wt=WT):
    def nd(a, b):
        return abs(a - b) / max(b, 1)

    d_gate = nd(len(test.data), len(ref.data))
    d_depth = nd(test.depth(), ref.depth())
    d_cx = nd(test.count_ops().get("cx", 0), ref.count_ops().get("cx", 0))

    v_ref = _dag_feature_vector(ref)
    v_tst = _dag_feature_vector(test)

    d_topo = float(np.sum(np.abs(v_tst - v_ref)) / (np.sum(np.abs(v_ref)) + 1e-9))

    sis = 1.0 - (wg * d_gate + wd * d_depth + wc * d_cx + wt * d_topo)

    return round(float(max(0.0, min(1.0, sis))), 6)


def compute_depth_only(ref, test):
    d = abs(test.depth() - ref.depth()) / max(ref.depth(), 1)
    return round(float(max(0.0, min(1.0, 1.0 - d))), 6)


def compute_gate_only(ref, test):
    d = abs(len(test.data) - len(ref.data)) / max(len(ref.data), 1)
    return round(float(max(0.0, min(1.0, 1.0 - d))), 6)


def compute_cnot_only(ref, test):
    rc = ref.count_ops().get("cx", 0)
    tc = test.count_ops().get("cx", 0)
    d = abs(tc - rc) / max(rc, 1)
    return round(float(max(0.0, min(1.0, 1.0 - d))), 6)


def compute_sis_no_topo(ref, test):
    return compute_sis(ref, test, wg=1/3, wd=1/3, wc=1/3, wt=0.0)


def compute_sis_no_cnot(ref, test):
    return compute_sis(ref, test, wg=1/3, wd=1/3, wc=0.0, wt=1/3)


# =============================================================================
# OIS
# =============================================================================

def _dist(counts, shots, keys):
    return np.array([counts.get(k, 0) / shots for k in keys], dtype=float)


def compute_ois(ref, test, cache, backend=None):
    if ref.num_qubits > MAX_QUBITS_OIS:
        return None, None, None, None

    bk = backend or BACKEND
    t0 = time.perf_counter()

    rm = ref.copy().measure_all(inplace=False)
    tm = test.copy().measure_all(inplace=False)

    tr = transpile(rm, bk, optimization_level=0)
    tt = transpile(tm, bk, optimization_level=0)

    if "counts" not in cache:
        cache["counts"] = bk.run(tr, shots=SHOTS).result().get_counts()

    cr = cache["counts"]
    ct = bk.run(tt, shots=SHOTS).result().get_counts()

    keys = sorted(set(cr) | set(ct))

    p = _dist(cr, SHOTS, keys)
    q = _dist(ct, SHOTS, keys)

    jsd = float(jensenshannon(p, q, base=2))
    ois = round(1.0 - jsd, 6)
    tvd = round(float(0.5 * np.sum(np.abs(p - q))), 6)

    dt = time.perf_counter() - t0

    return round(jsd, 6), ois, tvd, round(float(dt), 8)


def compute_fidelity(ref, test):
    if not QISKIT_AVAILABLE:
        return None

    if ref.num_qubits > MAX_QUBITS_FIDELITY:
        return None

    try:
        sv_ref = Statevector.from_instruction(ref)
        sv_tst = Statevector.from_instruction(test)
        return round(float(state_fidelity(sv_ref, sv_tst)), 6)
    except Exception:
        return None


# =============================================================================
# IGS-L
# =============================================================================

GATE_FAMILIES = [
    "h", "x", "y", "z", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "sx",
    "cx", "cz", "swap", "ccx",
    "measure", "reset", "other"
]

GATE_TO_IDX = {g: i for i, g in enumerate(GATE_FAMILIES)}

FINGERPRINT_DIM = len(GATE_FAMILIES) + 6 + 5

_UNITARY_CACHE = {}


def gate_family(name):
    n = name.lower()

    if n in GATE_TO_IDX:
        return n

    if n in ("u", "u1", "u2", "u3", "p", "id"):
        return "other"

    return "other"


def _gate_unitary_fingerprint(gate):
    try:
        key = (gate.name, tuple(float(p) for p in gate.params))
    except Exception:
        key = (gate.name,)

    if key in _UNITARY_CACHE:
        return _UNITARY_CACHE[key]

    try:
        U = Operator(gate).data
        n = U.shape[0]

        re = U.real
        im = U.imag

        tr_re = float(np.trace(re))
        dist_id = float(np.linalg.norm(U - np.eye(n)))
        im_mass = float(np.sum(np.abs(im)))
        max_re = float(np.max(np.abs(re)))
        max_im = float(np.max(np.abs(im)))
        off_diag = float(np.sum(np.abs(re - np.diag(np.diag(re)))))

        fp = np.array([tr_re, dist_id, im_mass, max_re, max_im, off_diag], dtype=float)

    except Exception:
        fp = np.zeros(6, dtype=float)

    _UNITARY_CACHE[key] = fp

    return fp


def node_feature_vector(qc, node):
    qids = tuple(qc.find_bit(q).index for q in node.qargs)
    qarr = np.array(qids, dtype=float) if len(qids) > 0 else np.array([0.0])

    fam = gate_family(node.name)

    one_hot = np.zeros(len(GATE_FAMILIES), dtype=float)
    one_hot[GATE_TO_IDX[fam]] = 1.0

    unitary_fp = _gate_unitary_fingerprint(node.op)

    qubit_stats = np.array([
        float(len(qids)),
        float(np.mean(qarr)),
        float(np.std(qarr)),
        float(np.min(qarr)),
        float(np.max(qarr)),
    ], dtype=float)

    return np.concatenate([one_hot, unitary_fp, qubit_stats])


def build_labeled_interaction_graph(qc):
    dag = circuit_to_dag(qc)
    G = nx.DiGraph()

    op_nodes = list(dag.topological_op_nodes())
    node_map = {}

    for idx, node in enumerate(op_nodes):
        node_map[node] = idx

        qids = tuple(qc.find_bit(q).index for q in node.qargs)
        feat = node_feature_vector(qc, node)

        G.add_node(idx, name=node.name, qargs=qids, features=feat)

    for node in op_nodes:
        src = node_map[node]

        for succ in dag.successors(node):
            if succ in node_map:
                dst = node_map[succ]
                G.add_edge(src, dst, edge_type="dag")

    qubit_to_nodes = defaultdict(list)

    for idx, node in enumerate(op_nodes):
        for q in node.qargs:
            qubit_to_nodes[qc.find_bit(q).index].append(idx)

    for _, node_indices in qubit_to_nodes.items():
        for a, b in zip(node_indices, node_indices[1:]):
            if not G.has_edge(a, b):
                qa = set(G.nodes[a]["qargs"])
                qb = set(G.nodes[b]["qargs"])
                shared = len(qa & qb)

                G.add_edge(a, b, edge_type="shared_qubit", shared_count=shared)

    return G


def graph_to_adjacency(G):
    n = G.number_of_nodes()

    if n == 0:
        return np.zeros((0, 0), dtype=float)

    return nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=float)


def graph_to_feature_matrix(G):
    n = G.number_of_nodes()

    if n == 0:
        return np.zeros((0, FINGERPRINT_DIM), dtype=float)

    rows = [G.nodes[idx]["features"] for idx in sorted(G.nodes())]

    return np.vstack(rows)


def pad_square_matrix(A, target_n):
    n = A.shape[0]

    if n == target_n:
        return A

    B = np.zeros((target_n, target_n), dtype=float)

    if n > 0:
        B[:n, :n] = A

    return B


def pad_feature_matrix(F, target_n):
    n, d = F.shape

    if n == target_n:
        return F

    B = np.zeros((target_n, d), dtype=float)

    if n > 0:
        B[:n, :] = F

    return B


def topology_difference(G_ref, G_tst):
    A_ref = graph_to_adjacency(G_ref)
    A_tst = graph_to_adjacency(G_tst)

    max_n = max(A_ref.shape[0], A_tst.shape[0])

    A_ref = pad_square_matrix(A_ref, max_n)
    A_tst = pad_square_matrix(A_tst, max_n)

    diff = float(np.sum(np.abs(A_tst - A_ref)))
    norm = float(np.sum(np.abs(A_ref)) + 1e-9)

    return diff / norm


def node_semantic_difference(G_ref, G_tst):
    F_ref = graph_to_feature_matrix(G_ref)
    F_tst = graph_to_feature_matrix(G_tst)

    max_n = max(F_ref.shape[0], F_tst.shape[0])

    F_ref = pad_feature_matrix(F_ref, max_n)
    F_tst = pad_feature_matrix(F_tst, max_n)

    diff = float(np.sum(np.abs(F_tst - F_ref)))
    norm = float(np.sum(np.abs(F_ref)) + 1e-9)

    return diff / norm


def order_difference(ref, test):
    seq_r = [gate_family(inst.operation.name) for inst in ref.data]
    seq_t = [gate_family(inst.operation.name) for inst in test.data]

    n = max(len(seq_r), len(seq_t))

    if n == 0:
        return 0.0

    mismatches = sum(
        0 if (seq_r[i] if i < len(seq_r) else "__PAD__") ==
             (seq_t[i] if i < len(seq_t) else "__PAD__")
        else 1
        for i in range(n)
    )

    return mismatches / n


def interaction_difference(ref, test):
    def interaction_multiset(qc):
        bag = Counter()

        for inst in qc.data:
            if len(inst.qubits) == 2:
                fam = gate_family(inst.operation.name)
                qids = tuple(sorted(qc.find_bit(q).index for q in inst.qubits))
                bag[(fam, qids)] += 1

        return bag

    bag_r = interaction_multiset(ref)
    bag_t = interaction_multiset(test)

    keys = set(bag_r.keys()).union(set(bag_t.keys()))

    diff = sum(abs(bag_r[k] - bag_t[k]) for k in keys)
    norm = sum(bag_r.values()) + 1e-9

    return diff / norm


def qubit_usage_difference(ref, test):
    n_q = max(ref.num_qubits, test.num_qubits)

    def usage_vector(qc):
        v = np.zeros(n_q, dtype=float)

        for inst in qc.data:
            for q in inst.qubits:
                v[qc.find_bit(q).index] += 1

        total = v.sum() + 1e-9

        return v / total

    uv_ref = usage_vector(ref)
    uv_tst = usage_vector(test)

    return float(np.sum(np.abs(uv_ref - uv_tst)))


def compute_igsl(ref, test, ref_graph=None):
    G_ref = ref_graph if ref_graph is not None else build_labeled_interaction_graph(ref)
    G_tst = build_labeled_interaction_graph(test)

    d_edge = min(1.0, topology_difference(G_ref, G_tst))
    d_node = min(1.0, node_semantic_difference(G_ref, G_tst))
    d_order = min(1.0, order_difference(ref, test))
    d_inter = min(1.0, interaction_difference(ref, test))
    d_usage = min(1.0, qubit_usage_difference(ref, test))

    delta_igsl = (
        W_EDGE * d_edge +
        W_NODE * d_node +
        W_ORDER * d_order +
        W_INTERACTION * d_inter +
        W_USAGE * d_usage
    )

    igsl = 1.0 - delta_igsl
    igsl = max(0.0, min(1.0, igsl))

    return (
        round(float(igsl), 6),
        round(float(d_edge), 6),
        round(float(d_node), 6),
        round(float(d_order), 6),
        round(float(d_inter), 6),
        round(float(d_usage), 6),
    )


def compute_igsl_with_time(ref, test, ref_graph=None):
    t0 = time.perf_counter()

    igsl, d_edge, d_node, d_order, d_inter, d_usage = compute_igsl(ref, test, ref_graph)

    dt = time.perf_counter() - t0

    return (
        igsl,
        d_edge,
        d_node,
        d_order,
        d_inter,
        d_usage,
        round(float(dt), 8),
    )


# =============================================================================
# ANOMALY INJECTION
# =============================================================================

def inject_fixed(qc, anomaly):
    q = qc.copy()

    if anomaly == "none":
        return q

    elif anomaly == "gate_deletion_1q":
        for i, inst in enumerate(q.data):
            if inst.operation.num_qubits == 1:
                q.data.pop(i)
                break

    elif anomaly == "gate_deletion_2q":
        for i, inst in enumerate(q.data):
            if inst.operation.num_qubits == 2:
                q.data.pop(i)
                break

    elif anomaly == "gate_insertion":
        if len(q.data) > 0:
            insert_pos = random.randint(0, len(q.data))
            target_qubit = random.randint(0, q.num_qubits - 1)
            q.data.insert(insert_pos, CircuitInstruction(HGate(), qubits=[q.qubits[target_qubit]]))

    elif anomaly == "gate_substitution":
        substitute_map = {
            "h": XGate(),
            "x": HGate(),
            "s": TGate(),
            "t": SGate(),
            "z": XGate(),
            "y": HGate(),
            "rx": HGate(),
            "ry": XGate(),
            "rz": SGate(),
            "sx": HGate(),
        }

        for i, inst in enumerate(q.data):
            op_name = inst.operation.name.lower()

            if inst.operation.num_qubits == 1 and op_name in substitute_map:
                q.data[i] = CircuitInstruction(substitute_map[op_name], qubits=inst.qubits)
                break

    elif anomaly == "gate_reorder":
        q.data = list(reversed(q.data))

    elif anomaly in ["trojan_NOT", "trojan_H"]:
        if q.num_qubits >= 1:
            qubit_gate_counts = [0] * q.num_qubits

            for inst in q.data:
                for qb in inst.qubits:
                    qubit_gate_counts[q.find_bit(qb).index] += 1

            idle_qubit = int(np.argmin(qubit_gate_counts))
            insert_pos = len(q.data)

            for i, inst in enumerate(q.data):
                qb_indices = [q.find_bit(qb).index for qb in inst.qubits]

                if idle_qubit not in qb_indices:
                    insert_pos = i
                    break

            gate = XGate() if anomaly == "trojan_NOT" else HGate()
            q.data.insert(insert_pos, CircuitInstruction(gate, qubits=[q.qubits[idle_qubit]]))

    elif anomaly == "qubit_swap":
        if q.num_qubits >= 2:
            q.swap(0, 1)

    return q


def inject_severity(qc, anomaly, severity):
    q = qc.copy()
    total = len(q.data)
    cap = max(1, total // 2)

    if anomaly == "none":
        return q

    elif anomaly == "gate_deletion_1q":
        elig = [i for i, inst in enumerate(q.data) if inst.operation.num_qubits == 1]

        if not elig:
            return q

        k = max(1, min(int(np.ceil(severity * len(elig))), len(elig), cap))
        rm = set(random.sample(elig, k))
        q.data = [inst for i, inst in enumerate(q.data) if i not in rm]

    elif anomaly == "gate_deletion_2q":
        elig = [i for i, inst in enumerate(q.data) if inst.operation.num_qubits == 2]

        if not elig:
            return q

        k = max(1, min(int(np.ceil(severity * len(elig))), len(elig), cap))
        rm = set(random.sample(elig, k))
        q.data = [inst for i, inst in enumerate(q.data) if i not in rm]

    elif anomaly == "gate_insertion":
        insert_gates = [HGate(), XGate(), SGate()]
        k = max(1, min(int(np.ceil(severity * total)), cap))

        for _ in range(k):
            pos = random.randint(0, len(q.data))
            qb = random.randint(0, q.num_qubits - 1)
            gate = random.choice(insert_gates)
            q.data.insert(pos, CircuitInstruction(gate, qubits=[q.qubits[qb]]))

    elif anomaly == "gate_substitution":
        substitute_map = {
            "h": XGate(),
            "x": HGate(),
            "s": TGate(),
            "t": SGate(),
            "z": XGate(),
            "y": HGate(),
            "sx": HGate(),
            "rx": HGate(),
            "ry": XGate(),
            "rz": SGate(),
        }

        elig = [
            i for i, inst in enumerate(q.data)
            if inst.operation.num_qubits == 1 and inst.operation.name.lower() in substitute_map
        ]

        if not elig:
            return q

        k = max(1, min(int(np.ceil(severity * len(elig))), len(elig)))
        chosen = random.sample(elig, k)

        for i in chosen:
            op_name = q.data[i].operation.name.lower()
            q.data[i] = CircuitInstruction(substitute_map.get(op_name, HGate()), qubits=q.data[i].qubits)

    elif anomaly == "gate_reorder":
        d = list(q.data)
        n_swap = max(1, int(np.ceil(severity * len(d))))

        for _ in range(n_swap):
            if len(d) >= 2:
                i, j = random.sample(range(len(d)), 2)
                d[i], d[j] = d[j], d[i]

        q.data = d

    elif anomaly in ["trojan_NOT", "trojan_H"]:
        k = max(1, min(int(np.ceil(severity * total)), cap))
        gate = XGate() if anomaly == "trojan_NOT" else HGate()

        for _ in range(k):
            qb = random.randint(0, q.num_qubits - 1)
            pos = random.randint(0, len(q.data))
            q.data.insert(pos, CircuitInstruction(gate, qubits=[q.qubits[qb]]))

    elif anomaly == "qubit_swap":
        k = max(1, min(int(np.ceil(severity * total)), cap))

        if q.num_qubits >= 2:
            for _ in range(k):
                a, b = random.sample(range(q.num_qubits), 2)
                q.swap(a, b)

    return q


# =============================================================================
# CIRCUIT LOADING
# =============================================================================

def load_filtered_circuits(root_folder):
    circuits = []
    skipped = []

    for size_cat in ["small", "medium", "large"]:
        folder = os.path.join(root_folder, size_cat)

        if not os.path.exists(folder):
            continue

        for r, _, files in os.walk(folder):
            for f in files:
                if not f.endswith(".qasm"):
                    continue

                path = os.path.join(r, f)

                try:
                    qc = QuantumCircuit.from_qasm_file(path)

                except Exception as e:
                    skipped.append({
                        "name": f,
                        "size_category": size_cat,
                        "reason": f"parse_error:{str(e)[:100]}"
                    })
                    continue

                if qc.num_qubits > MAX_QUBITS_STRUCT:
                    skipped.append({
                        "name": f,
                        "size_category": size_cat,
                        "reason": f"too_many_qubits({qc.num_qubits})"
                    })
                    continue

                if len(qc.data) > MAX_GATES:
                    skipped.append({
                        "name": f,
                        "size_category": size_cat,
                        "reason": f"too_many_gates({len(qc.data)})"
                    })
                    continue

                circuits.append((f, qc, size_cat))

    print(f"Loaded {len(circuits)} circuits | Skipped {len(skipped)}")

    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(OUTPUT_DIR, "skipped_circuits_log.csv"), index=False)

    return circuits


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark():
    fixed_rows = []
    sev_rows = []

    circuits = load_filtered_circuits(ROOT_FOLDER)

    for name, ref, size_cat in tqdm(circuits, desc="Benchmarking"):
        base = {
            "name": name,
            "size_category": size_cat,
            "family": get_family(name),
            "qubits": ref.num_qubits,
            "gates": len(ref.data),
        }

        cache = {}
        ref_graph = build_labeled_interaction_graph(ref)

        for anom in ANOMALY_TYPES:
            test = ref.copy() if anom == "none" else inject_fixed(ref, anom)

            jsd, ois, tvd, ois_time = compute_ois(ref, test, cache)
            fid = compute_fidelity(ref, test)

            sis_full = 1.0 if anom == "none" else compute_sis(ref, test)

            igsl, d_edge, d_node, d_order, d_inter, d_usage, igsl_time = compute_igsl_with_time(
                ref, test, ref_graph=ref_graph
            )

            fixed_rows.append({
                **base,
                "mode": "fixed",
                "anomaly": anom,
                "severity": "fixed",
                "SIS": sis_full,
                "IGSL": igsl,
                "IGSL_edge_diff": d_edge,
                "IGSL_node_diff": d_node,
                "IGSL_order_diff": d_order,
                "IGSL_interaction_diff": d_inter,
                "IGSL_usage_diff": d_usage,
                "IGSL_time_sec": igsl_time,
                "SIS_depth_only": compute_depth_only(ref, test),
                "SIS_gate_only": compute_gate_only(ref, test),
                "SIS_cnot_only": compute_cnot_only(ref, test),
                "SIS_no_topo": 1.0 if anom == "none" else compute_sis_no_topo(ref, test),
                "SIS_no_cnot": 1.0 if anom == "none" else compute_sis_no_cnot(ref, test),
                "JSD_dist": jsd,
                "OIS_sim": ois,
                "OIS_time_sec": ois_time,
                "TVD": tvd,
                "Fidelity": fid,
                "primary_sis_component": ANOMALY_SIS_COMPONENT.get(anom, "all"),
            })

        for anom in [a for a in ANOMALY_TYPES if a != "none"]:
            for sev in SEVERITY_LEVELS:
                test = inject_severity(ref, anom, sev)

                jsd, ois, tvd, ois_time = compute_ois(ref, test, cache)

                igsl, d_edge, d_node, d_order, d_inter, d_usage, igsl_time = compute_igsl_with_time(
                    ref, test, ref_graph=ref_graph
                )

                sev_rows.append({
                    **base,
                    "mode": "severity",
                    "anomaly": anom,
                    "severity": sev,
                    "SIS": compute_sis(ref, test),
                    "IGSL": igsl,
                    "IGSL_edge_diff": d_edge,
                    "IGSL_node_diff": d_node,
                    "IGSL_order_diff": d_order,
                    "IGSL_interaction_diff": d_inter,
                    "IGSL_usage_diff": d_usage,
                    "IGSL_time_sec": igsl_time,
                    "JSD_dist": jsd,
                    "OIS_sim": ois,
                    "OIS_time_sec": ois_time,
                    "TVD": tvd,
                })

    df_f = pd.DataFrame(fixed_rows)
    df_s = pd.DataFrame(sev_rows)

    df_f.to_csv(os.path.join(OUTPUT_DIR, "benchmark_fixed.csv"), index=False)
    df_s.to_csv(os.path.join(OUTPUT_DIR, "benchmark_severity.csv"), index=False)

    print("Benchmark complete.")

    return df_f, df_s


# =============================================================================
# PLOTS
# =============================================================================

def make_igsl_output_plots(df_f, df_s):
    print("\nGenerating IGS-L output plots")

    df = df_f.copy()
    df = df[df["anomaly"] != "none"].copy()
    df_valid = df.dropna(subset=["SIS", "IGSL", "OIS_sim"]).copy()

    sis_fail_threshold = 0.95
    df_fail = df_valid[df_valid["SIS"] >= sis_fail_threshold].copy()

    if len(df_fail) > 0:
        sens = (
            df_fail.groupby("anomaly")
            .agg(
                mean_SIS=("SIS", "mean"),
                mean_IGSL=("IGSL", "mean"),
                mean_OIS=("OIS_sim", "mean"),
                count=("SIS", "count"),
            )
            .reset_index()
        )

        sens = sens[sens["anomaly"].isin(ANOMALY_ORDER)]
        sens["anomaly"] = pd.Categorical(sens["anomaly"], categories=ANOMALY_ORDER, ordered=True)
        sens = sens.sort_values("anomaly")

        sens_long = sens.melt(
            id_vars=["anomaly", "count"],
            value_vars=["mean_SIS", "mean_IGSL", "mean_OIS"],
            var_name="Metric",
            value_name="Value"
        )

        sens_long["Metric"] = sens_long["Metric"].map({
            "mean_SIS": "SIS",
            "mean_IGSL": "IGS-L",
            "mean_OIS": "OIS",
        })

        plt.figure(figsize=(13, 6))
        ax = sns.barplot(data=sens_long, x="anomaly", y="Value", hue="Metric")

        ax.set_title(f"Metric Response When SIS Remains High (SIS >= {sis_fail_threshold})")
        ax.set_xlabel("Anomaly type")
        ax.set_ylabel("Mean score")
        ax.set_ylim(0, 1.05)

        plt.xticks(rotation=35, ha="right")

        for i, row in enumerate(sens.itertuples(index=False)):
            ax.text(i, 1.01, f"n={row.count}", ha="center", va="bottom", fontsize=8)

        save("Fig_IGSL_sensitivity_when_SIS_high.png")
        sens.to_csv(os.path.join(OUTPUT_DIR, "igsl_sensitivity_when_sis_high.csv"), index=False)

    time_df = (
        df_valid.groupby("qubits")
        .agg(
            mean_IGSL_time=("IGSL_time_sec", "mean"),
            std_IGSL_time=("IGSL_time_sec", "std"),
            mean_OIS_time=("OIS_time_sec", "mean"),
            std_OIS_time=("OIS_time_sec", "std"),
            n=("IGSL_time_sec", "count"),
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 5))

    plt.errorbar(
        time_df["qubits"],
        time_df["mean_IGSL_time"],
        yerr=time_df["std_IGSL_time"].fillna(0),
        marker="o",
        linewidth=2,
        capsize=4,
        label="IGS-L"
    )

    plt.errorbar(
        time_df["qubits"],
        time_df["mean_OIS_time"],
        yerr=time_df["std_OIS_time"].fillna(0),
        marker="s",
        linewidth=2,
        capsize=4,
        label="OIS"
    )

    plt.title("Runtime Comparison Between IGS-L and OIS")
    plt.xlabel("Qubit count")
    plt.ylabel("Runtime (seconds)")
    plt.legend()

    save("Fig_runtime_IGSL_vs_OIS_vs_qubits.png")
    time_df.to_csv(os.path.join(OUTPUT_DIR, "runtime_igsl_vs_ois_vs_qubits.csv"), index=False)


def make_summary_tables(df_f, df_s):
    print("\nGenerating summary tables")

    df_a = df_f[df_f["anomaly"] != "none"].copy()
    df_o = df_a.dropna(subset=["OIS_sim"]).copy()

    rows = []

    for anom in ANOMALY_ORDER:
        sub = df_a[df_a["anomaly"] == anom]
        sub2 = df_o[df_o["anomaly"] == anom]

        row = {
            "anomaly": anom,
            "primary_sis_component": ANOMALY_SIS_COMPONENT.get(anom, "all"),
            "n_circuits": len(sub),
            "mean_SIS": round(sub["SIS"].mean(), 4),
            "std_SIS": round(sub["SIS"].std(), 4),
            "mean_IGSL": round(sub["IGSL"].dropna().mean(), 4),
            "std_IGSL": round(sub["IGSL"].dropna().std(), 4),
            "mean_IGSL_time_sec": round(sub["IGSL_time_sec"].dropna().mean(), 6),
        }

        for col in [
            "IGSL_edge_diff",
            "IGSL_node_diff",
            "IGSL_order_diff",
            "IGSL_interaction_diff",
            "IGSL_usage_diff",
        ]:
            vals = sub[col].dropna()

            if len(vals) > 0:
                row[f"mean_{col}"] = round(vals.mean(), 4)

        if not sub2.empty:
            row["mean_OIS"] = round(sub2["OIS_sim"].mean(), 4)
            row["std_OIS"] = round(sub2["OIS_sim"].std(), 4)
            row["mean_OIS_time_sec"] = round(sub2["OIS_time_sec"].dropna().mean(), 6)

        rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "benchmark_summary.csv"), index=False)

    print(f"Saved -> {os.path.join(OUTPUT_DIR, 'benchmark_summary.csv')}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 74)
    print("FINAL BENCHMARK - SIS + OIS + INDEPENDENT IGS-L")
    print("=" * 74)

    print("\nIGS-L weights used:")
    print(f"  W_EDGE        = {W_EDGE:.2f}")
    print(f"  W_NODE        = {W_NODE:.2f}")
    print(f"  W_ORDER       = {W_ORDER:.2f}")
    print(f"  W_INTERACTION = {W_INTERACTION:.2f}")
    print(f"  W_USAGE       = {W_USAGE:.2f}")
    print(f"  Total         = {W_EDGE + W_NODE + W_ORDER + W_INTERACTION + W_USAGE:.2f}")

    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Qiskit is not available. Install qiskit and qiskit-aer before running this benchmark."
        )

    print("\nMode: REAL benchmark")
    df_fixed, df_sev = run_benchmark()

    make_igsl_output_plots(df_fixed, df_sev)
    make_summary_tables(df_fixed, df_sev)

    print("\nDONE")
    print("=" * 74)
    print("Outputs:")
    print("  data/benchmark_fixed.csv")
    print("  data/benchmark_severity.csv")
    print("  data/benchmark_summary.csv")
    print("  data/igsl_sensitivity_when_sis_high.csv")
    print("  data/runtime_igsl_vs_ois_vs_qubits.csv")
    print("  plots/Fig_IGSL_sensitivity_when_SIS_high.png")
    print("  plots/Fig_runtime_IGSL_vs_OIS_vs_qubits.png")