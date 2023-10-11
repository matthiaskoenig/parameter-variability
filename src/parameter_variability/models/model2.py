"""Simple model for demonstration."""
from pathlib import Path

from sbmlutils.console import console
from sbmlutils.cytoscape import visualize_sbml
from sbmlutils.factory import *
from sbmlutils.metadata import *


_m = Model(
    sid="model2",
    name="Model 2",
)
_m.compartments = [
    Compartment(
        sid="liver",
        value=1.0,
    )
]

_m.species = [
    Species(
        sid="gut",
        name="gut",
        compartment='liver',
        initialConcentration=1.0,
    ),
    Species(
        sid="cent",
        name='cent',
        compartment="liver",
        initialConcentration=0.0,
    ),
    Species(
        sid='peri',
        name='peri',
        compartment='liver',
        initialConcentration=0.0
    )
]

_m.parameters = [
    Parameter(
        sid="k",
        value=1.0
    ),
    Parameter(
        sid="cl",
        value=1.0
    ),
    Parameter(
        sid="q",
        value=1.0
    ),
    Parameter(
        sid="v_cent",
        value=1.0
    ),
    Parameter(
        sid="v_peri",
        value=1.0
    )
]

_m.reactions = [
    Reaction(
        sid="R1",
        equation="gut -> gut",
        formula="-k * gut",
        notes="""
        dgut /dt = - k * gut
        """
    ),
    Reaction(
        sid="R2",
        equation="gut, cent, peri -> cent",
        formula="k * gut - (cl/v_cent + q/v_cent)*cent + q/v_peri*peri",
        notes="""
        dcent/dt = k * gut - (cl/v_cent + q/v_cent)*cent + q/v_peri*peri
        """
    ),
    Reaction(
        sid="R3",
        equation="cent, peri -> peri",
        formula="q/v_cent*cent - q/v_peri*peri",
        notes="""
        dperi /dt = q/v_cent*cent - q/v_peri*peri
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
    visualize_sbml(sbml_path=results.sbml_path)
