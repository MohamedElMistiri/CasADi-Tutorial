"""Load and evaluate the compiled function produced by codegen_smoke.py."""

import sys

import casadi as ca


def main(library: str) -> None:
    """Load a generated shared library and validate its two output matrices."""
    # ``external`` resolves the C symbols emitted by ``Function.generate``.
    function = ca.external("generated_dynamics", library)
    value, jacobian = function(ca.DM.zeros(4), ca.DM.ones(4))
    assert value.shape == (4, 1)
    assert jacobian.shape == (4, 4)
    assert float(ca.norm_inf(value)) == 0.0


if __name__ == "__main__":
    # The workflow passes the platform-specific shared-library path explicitly.
    main(sys.argv[1])
