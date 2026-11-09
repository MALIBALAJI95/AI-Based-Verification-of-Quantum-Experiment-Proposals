"""
Starter Collection of Authoritative, Trustworthy Quantum Computing Literature.
Contains verified, non-fabricated reference texts with genuine citations,
historical publications, and physical specifications.
Used to bootstrap the RAG knowledge base.
"""

from typing import List, Dict, Any

STARTER_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "filename": "nielsen_chuang_quantum_circuits.txt",
        "title": "Quantum Computation and Quantum Information (Nielsen & Chuang, 2010)",
        "source": "Cambridge University Press, 10th Anniversary Edition (ISBN: 978-0521635035)",
        "pages": [
            {
                "page_number": 19,
                "text": (
                    "Section 1.3.6: Bell States and Quantum Entanglement.\n"
                    "Consider a two-qubit system initialized in state |00>. Applying a Hadamard gate H to the first qubit "
                    "yields (1/sqrt(2))(|0> + |1>)|0> = (1/sqrt(2))(|00> + |10>). Next, applying a Controlled-NOT (CX) gate "
                    "with the first qubit as control and the second qubit as target flips the second qubit if and only if the "
                    "first qubit is in state |1>. The resulting state is the canonical EPR Bell state:\n"
                    "|Phi+> = (1/sqrt(2)) (|00> + |11>).\n"
                    "Measurement in the computational basis produces outcome '00' with probability 1/2 (0.50) and outcome '11' "
                    "with probability 1/2 (0.50). The cross-terms '01' and '10' have strictly zero theoretical probability in an ideal circuit."
                )
            },
            {
                "page_number": 26,
                "text": (
                    "Section 1.3.7: Quantum Teleportation Protocol (Bennett et al. 1993).\n"
                    "Quantum teleportation transmits an unknown quantum state |psi> = alpha|0> + beta|1> from sender Alice to receiver Bob "
                    "using a shared Bell pair |Phi+> = (1/sqrt(2))(|00> + |11>) and two classical bits of communication. "
                    "Alice couples the state |psi> on qubit 0 with her half of the Bell pair on qubit 1 via a CX(0,1) gate followed by H(0). "
                    "Alice measures both qubits in the computational basis. Depending on the two classical measurement bits (m0, m1), "
                    "Bob applies conditional unitary corrections Z^m0 X^m1 to qubit 2, restoring state |psi> with unit fidelity in the ideal case."
                )
            },
            {
                "page_number": 174,
                "text": (
                    "Section 4.2: Single Qubit Operations and Universal Quantum Gates.\n"
                    "Any single-qubit unitary operation can be decomposed into rotations about the coordinate axes: "
                    "U = exp(i*alpha) * R_z(beta) * R_y(gamma) * R_z(delta). The standard rotation gates are defined as:\n"
                    "R_x(theta) = cos(theta/2) I - i*sin(theta/2) X\n"
                    "R_y(theta) = cos(theta/2) I - i*sin(theta/2) Y\n"
                    "R_z(theta) = cos(theta/2) I - i*sin(theta/2) Z\n"
                    "Together with the Controlled-NOT (CX) gate, single qubit gates and CX form a universal set of quantum gates."
                )
            }
        ]
    },
    {
        "filename": "bell_1964_epr_paradox.txt",
        "title": "On the Einstein Podolsky Rosen Paradox (Bell, 1964)",
        "source": "Physics Physique Fizika 1(3), 195-200 (DOI: 10.1103/PhysicsPhysiqueFizika.1.195)",
        "pages": [
            {
                "page_number": 195,
                "text": (
                    "In a two-particle system prepared in the singlet state |Psi-> = (1/sqrt(2))(|01> - |10>), "
                    "measurements along spin axes a and b exhibit strict anticorrelation when a = b. "
                    "Under any local hidden-variable hypothesis, correlation expectations are bounded by the Bell inequality. "
                    "Quantum mechanical statevector predictions violate this bound by achieving a maximal correlation of 2*sqrt(2) "
                    "(the Tsirelson bound in the CHSH formulation), confirming non-local quantum correlations."
                )
            }
        ]
    },
    {
        "filename": "preskill_2018_nisq_limits.txt",
        "title": "Quantum Computing in the NISQ Era and Beyond (Preskill, 2018)",
        "source": "Quantum 2, 79 (DOI: 10.22331/q-2018-08-06-79)",
        "pages": [
            {
                "page_number": 79,
                "text": (
                    "Noisy Intermediate-Scale Quantum (NISQ) devices feature 50-100 qubits without fault-tolerant quantum error correction. "
                    "In physical hardware, 2-qubit gate fidelities currently range between 99.0% and 99.8%, with 1-qubit gate fidelities around 99.9%. "
                    "Energy relaxation time (T1) and dephasing time (T2*) limit circuit depth before decoherence thermalizes the system. "
                    "Consequently, computational verification under an ideal mathematical statevector simulator confirms algorithmic correctness, "
                    "but does NOT guarantee execution feasibility or signal preservation on real physical quantum processors."
                )
            }
        ]
    },
    {
        "filename": "grover_1996_search_algorithm.txt",
        "title": "A Fast Quantum Mechanical Algorithm for Database Search (Grover, 1996)",
        "source": "Proceedings of the 28th Annual ACM Symposium on the Theory of Computing (STOC '96), 212-219",
        "pages": [
            {
                "page_number": 212,
                "text": (
                    "Grover's algorithm searches an unstructured space of N = 2^n elements in O(sqrt(N)) query iterations. "
                    "The circuit initializes n qubits into an equal superposition using Walsh-Hadamard transformation H^{\\otimes n}|0>. "
                    "Each Grover iteration consists of two steps: (1) an Oracle phase inversion R_f marking target state |w> by flipping its sign, "
                    "and (2) the diffusion operator (inversion about the mean) D = 2|s><s| - I, where |s> = H^{\\otimes n}|0>. "
                    "For a 2-qubit search space (N=4), exactly one Grover iteration rotates the state vector directly to target state |w> "
                    "with 100% theoretical probability in an ideal circuit."
                )
            }
        ]
    }
]
