"""Simple model for demonstration."""
from pathlib import Path

from sbmlutils.console import console
from sbmlutils.converters import odefac
from sbmlutils.cytoscape import visualize_sbml
from sbmlutils.factory import *
from sbmlutils.metadata import *

_m = Model(
    sid="simple_chain",
    name="Model Simple Chain",
    notes="""Simple S1 -> S2 conversion for testing.""",
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
    ),
]

_m.parameters = [Parameter(sid="k1", value=1.0)]

_m.reactions = [
    Reaction(
        sid="R1",
        equation="S1 -> S2",
        formula="k1 * S1",
        notes="""
        dS1 /dt = - k1 * S1
        dS2 /dt = + k1 * S1
        """,
    )
]


if __name__ == "__main__":
    from parameter_variability import MODELS_DIR

    results: FactoryResult = create_model(
        model=_m,
        filepath=MODELS_DIR / f"{_m.sid}.xml",
        sbml_level=3,
        sbml_version=2,
        validation_options=ValidationOptions(units_consistency=False),
    )

    # create differential equations
    md_path = MODELS_DIR / f"{_m.sid}.md"
    ode_factory = odefac.SBML2ODE.from_file(sbml_file=results.sbml_path)
    ode_factory.to_markdown(md_file=md_path)

    console.rule(style="white")
    from rich.markdown import Markdown

    with open(md_path, "r") as f:
        md_str = f.read()
        md = Markdown(md_str)
        console.print(md)
    console.rule(style="white")

    visualize_sbml(sbml_path=results.sbml_path)
