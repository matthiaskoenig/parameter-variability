"""Simple model for demonstration."""
from pathlib import Path

from sbmlutils.console import console
from sbmlutils.converters import odefac
from sbmlutils.cytoscape import visualize_sbml
from sbmlutils.factory import *
from sbmlutils.metadata import *


_m = Model(
    sid="model2",
    name="Model 2",
)
_m.compartments = [
    Compartment(
        sid="Vgut",
        name="gut compartment",
        value=1.0,
    ),
    Compartment(
        sid="Vperi",
        name="peripheral compartment",
        value=1.0,
    ),
    Compartment(
        sid="Vcent",
        name="central compartment",
        value=1.0,
    )
]

_m.species = [
    Species(
        sid="y_gut",
        name="y gut",
        compartment='Vgut',
        initialAmount=1.0,
        hasOnlySubstanceUnits=False,
        notes="""
        handled in amount not concentration
        """
    ),
    Species(
        sid="y_cent",
        name='y central',
        compartment="Vcent",
        initialConcentration=0.0,
    ),
    Species(
        sid='y_peri',
        name='y peripheral',
        compartment='Vperi',
        initialConcentration=0.0
    )
]

_m.parameters = [
    Parameter(
        sid="k",
        value=1.0
    ),
    Parameter(
        sid="CL",
        value=1.0
    ),
    Parameter(
        sid="Q",
        value=1.0
    ),
]

_m.reactions = [
    Reaction(
        sid="ABSORPTION",
        name="absorption",
        equation="y_gut -> y_cent",
        formula="-k * y_gut",
        notes="""
        absorption from gut
        """
    ),
    Reaction(
        sid="CLEARANCE",
        name="clearance",
        equation="y_cent ->",
        formula="-CL * y_cent",
        notes="""
        clearance from central compartment
        """
    ),

    Reaction(
        sid="R1",
        equation="y_cent -> y_peri",
        formula="Q* y_cent",
        notes="""
        distribution in peripheral compartment
        """
    ),
    Reaction(
        sid="R2",
        equation="y_peri -> y_cent",
        formula="Q * y_peri",
        notes="""
        distribution from peripheral compartment
        """
    )

]


if __name__ == "__main__":

    results: FactoryResult = create_model(
        model=_m,
        filepath=Path(__file__).parent / f"{_m.sid}.xml",
        sbml_level=3,
        sbml_version=2,
        validation_options=ValidationOptions(units_consistency=False)
    )
    # create differential equations
    md_path = Path(__file__).parent / f"{_m.sid}.md"
    ode_factory = odefac.SBML2ODE.from_file(sbml_file=results.sbml_path)
    ode_factory.to_markdown(md_file=md_path)

    console.rule(style="white")
    from rich.markdown import Markdown
    with open(md_path, "r") as f:
        md_str = f.read()
        md = Markdown(md_str)
        console.print(md)
    console.rule(style="white")

    # visualize network
    visualize_sbml(sbml_path=results.sbml_path, delete_session=True)
