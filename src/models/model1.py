"""Simple model for demonstration."""
from pathlib import Path

from sbmlutils.console import console
from sbmlutils.cytoscape import visualize_sbml
from sbmlutils.factory import *
from sbmlutils.metadata import *


_m = Model(
    sid="model1",
    name="Model 1",
)
_m.compartments = [
    Compartment(
        sid="liver",
        value=1.0,
    )
]

_m.species = [
    Species(
        sid="S1",
        name="S1",
        compartment="liver",
        initialConcentration=1.0,
    ),
    Species(
        sid="S2",
        name="S2",
        compartment="liver",
        initialConcentration=0.0,
    )
]

_m.parameters = [
    Parameter(
        sid="k1",
        value=1.0
    )
]

_m.reactions = [
    Reaction(
        sid="R1",
        equation="S1 -> S2",
        formula="k1 * S1",
        notes="""
        dS1 /dt = - k1 * S1
        dS2 /dt = + k1 * S1
        """

    )
]


if __name__ == "__main__":
    model_path = Path(__file__).parent / "model1.xml"

    results: FactoryResult = create_model(
        model=_m,
        filepath=model_path,
        sbml_level=3,
        sbml_version=2,
        validation_options=ValidationOptions(units_consistency=False)
    )
    visualize_sbml(sbml_path=results.sbml_path)
