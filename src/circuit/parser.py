"""
Quantum Circuit Parser and Validator.
Converts text, JSON, or DSL representations into validated CircuitSpec models.
Enforces gate vocabulary, qubit arity, qubit index bounds, and parameter constraints.
"""

import re
import json
from typing import List, Dict, Any, Union, Optional
from ..schemas.proposal import CircuitSpec, GateOperation

# Supported gate set for the verification prototype
SUPPORTED_SINGLE_QUBIT_GATES = {
    "H", "X", "Y", "Z", "S", "T", "SDG", "TDG", "I", "ID", "RX", "RY", "RZ"
}
SUPPORTED_TWO_QUBIT_GATES = {
    "CX", "CNOT", "CZ", "SWAP"
}
ALL_SUPPORTED_GATES = SUPPORTED_SINGLE_QUBIT_GATES.union(SUPPORTED_TWO_QUBIT_GATES)


class CircuitParseError(ValueError):
    """Raised when circuit syntax or specification cannot be parsed or validated."""
    pass


def normalize_gate_name(gate_name: str) -> str:
    """Normalize aliases such as CNOT -> CX, ID -> I."""
    g = gate_name.strip().upper()
    if g == "CNOT":
        return "CX"
    if g == "ID":
        return "I"
    return g


def parse_gate_line(line: str, line_num: int = 1) -> GateOperation:
    """
    Parse a single line describing a gate operation.
    Supported syntaxes:
      - 'H 0' or 'H q[0]'
      - 'CX 0 1' or 'CNOT 0, 1' or 'CX q[0], q[1]'
      - 'RZ(1.5708) 0' or 'RX(pi/2) 0'
    """
    clean = line.split('#')[0].strip()
    if not clean:
        raise CircuitParseError(f"Line {line_num}: Empty gate line.")

    # Check for parametric gate, e.g. RZ(1.5708) or RX(0.785)
    param_match = re.match(r"^([A-Za-z]+)\s*\(([^)]+)\)\s*(.*)$", clean)
    params: Optional[List[float]] = None
    if param_match:
        raw_name = param_match.group(1).strip().upper()
        raw_param_str = param_match.group(2).strip()
        rest = param_match.group(3).strip()
        
        # Parse expression (support float numbers and simple fractions of pi)
        try:
            val_str = raw_param_str.lower().replace("pi", "3.141592653589793")
            # Safe eval of basic arithmetic like 3.14159/2
            allowed_chars = set("0123456789+-*/. ")
            if not all(c in allowed_chars for c in val_str):
                raise ValueError(f"Unsafe or invalid characters in parameter: {raw_param_str}")
            param_val = float(eval(val_str, {"__builtins__": {}}, {}))
            params = [param_val]
        except Exception as e:
            raise CircuitParseError(f"Line {line_num}: Invalid gate parameter '{raw_param_str}': {e}")
        qubit_tokens_str = rest
    else:
        # Non-parametric gate, e.g., H 0 or CX 0 1
        parts = clean.split(maxsplit=1)
        raw_name = parts[0].strip().upper()
        qubit_tokens_str = parts[1].strip() if len(parts) > 1 else ""

    norm_name = normalize_gate_name(raw_name)
    if norm_name not in ALL_SUPPORTED_GATES:
        raise CircuitParseError(
            f"Line {line_num}: Unsupported gate '{raw_name}'. "
            f"Supported gates are: {sorted(list(ALL_SUPPORTED_GATES))}"
        )

    # Extract integer qubit indices from qubit_tokens_str
    # Extract all integer digits (e.g. '0', 'q[0]', 'q0')
    qubit_matches = re.findall(r"\b\d+\b", qubit_tokens_str)
    if not qubit_matches:
        raise CircuitParseError(
            f"Line {line_num}: No target qubit specified for gate '{norm_name}' in line: '{line}'"
        )
    target_qubits = [int(m) for m in qubit_matches]

    # Validate arity
    if norm_name in SUPPORTED_SINGLE_QUBIT_GATES and len(target_qubits) != 1:
        raise CircuitParseError(
            f"Line {line_num}: Single-qubit gate '{norm_name}' requires exactly 1 target qubit, but got {len(target_qubits)}."
        )

    if norm_name in SUPPORTED_TWO_QUBIT_GATES and len(target_qubits) != 2:
        raise CircuitParseError(
            f"Line {line_num}: Two-qubit gate '{norm_name}' requires exactly 2 qubits (control, target), but got {len(target_qubits)}."
        )

    if norm_name in SUPPORTED_TWO_QUBIT_GATES and target_qubits[0] == target_qubits[1]:
        raise CircuitParseError(
            f"Line {line_num}: Control and target qubits for two-qubit gate '{norm_name}' cannot be identical: {target_qubits[0]}."
        )

    return GateOperation(
        name=norm_name,
        target_qubits=target_qubits,
        params=params,
        description=f"Operation on qubit(s) {target_qubits}"
    )


def parse_circuit_dsl(text: str, num_qubits: Optional[int] = None) -> CircuitSpec:
    """
    Parse a multiline string of gate operations.
    Example:
        H 0
        CX 0 1
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise CircuitParseError("Circuit specification text is empty. Provide at least one gate operation.")

    parsed_gates: List[GateOperation] = []
    max_qubit_found = -1

    for idx, line in enumerate(lines, start=1):
        # Ignore comments or special metadata headers
        if line.lower().startswith("qubits:") or line.lower().startswith("num_qubits:"):
            header_num = int(re.findall(r"\d+", line)[0])
            if num_qubits is None:
                num_qubits = header_num
            continue

        gate = parse_gate_line(line, line_num=idx)
        parsed_gates.append(gate)
        for q in gate.target_qubits:
            if q > max_qubit_found:
                max_qubit_found = q

    if num_qubits is None:
        num_qubits = max(1, max_qubit_found + 1)
    else:
        # Validate that no gate uses an index >= num_qubits
        if max_qubit_found >= num_qubits:
            raise CircuitParseError(
                f"Gate references qubit index {max_qubit_found}, but circuit is specified with only {num_qubits} qubits (valid indices: 0..{num_qubits - 1})."
            )

    return CircuitSpec(num_qubits=num_qubits, gates=parsed_gates)


def validate_and_load_circuit(data: Union[CircuitSpec, Dict[str, Any], str], default_num_qubits: Optional[int] = None) -> CircuitSpec:
    """
    Unified entry point to parse and validate a circuit from:
    - An existing CircuitSpec object
    - A dictionary representation
    - A raw DSL string
    """
    if isinstance(data, CircuitSpec):
        return data

    if isinstance(data, dict):
        # Could be serialized CircuitSpec
        return CircuitSpec(**data)

    if isinstance(data, str):
        # Try JSON first
        trimmed = data.strip()
        if trimmed.startswith("{"):
            try:
                parsed_json = json.loads(trimmed)
                return CircuitSpec(**parsed_json)
            except Exception:
                pass
        return parse_circuit_dsl(data, num_qubits=default_num_qubits)

    raise CircuitParseError(f"Unsupported circuit specification type: {type(data)}")
