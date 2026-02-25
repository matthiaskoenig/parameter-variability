"""Simple model for demonstration."""

from sbmlutils.converters import odefac
from sbmlutils.cytoscape import visualize_sbml
from sbmlutils.examples.templates import terms_of_use
from sbmlutils.factory import *
from sbmlutils.metadata import *


class U(Units):
    """UnitDefinitions."""

    mmole = UnitDefinition("mmole")
    min = UnitDefinition("min")
    mg = UnitDefinition("mg")
    per_min = UnitDefinition("per_min", "1/min")
    per_min_l = UnitDefinition("per_min_l", "1/min/liter")
    m2 = UnitDefinition("m2", "meter^2")
    mM = UnitDefinition("mM", "mmole/liter")
    mmole_per_min = UnitDefinition("mmole_per_min", "mmole/min")
    mmole_per_min_l = UnitDefinition("mmole_per_min_l", "mmole/min/liter")
    g_per_mole = UnitDefinition("g_per_mole", "g/mole")
    l_per_min = UnitDefinition("l_per_min", "l/min")
    l_per_min_mmole = UnitDefinition("l_per_min_mmole", "l/min/mmole")
    mM = UnitDefinition("mM", "mmole/liter")


_m = Model(
    sid="simple_pk",
    name="Simple PK model",
    notes="""
    # Model of absorption and distribution of substance y.
    """
    + terms_of_use,
    creators=[
        Creator(
            familyName="Alvarez",
            givenName="Antonio",
            email="antonio.alvarez@student.hu-berlin.de",
            organization="Humboldt-University Berlin, Institute for Theoretical Biology",
        ),
        Creator(
            familyName="König",
            givenName="Matthias",
            email="koenigmx@hu-berlin.de",
            organization="Humboldt-University Berlin, Institute for Theoretical Biology",
            site="https://livermetabolism.com",
        ),
    ],
    units=U,
    model_units=ModelUnits(
        time=U.min,
        extent=U.mmole,
        substance=U.mmole,
        length=U.meter,
        area=U.m2,
        volume=U.liter,
    ),
)
_m.compartments = [
    Compartment(
        sid="Vgut",
        name="gut compartment",
        value=1.0,
        sboTerm=SBO.PHYSICAL_COMPARTMENT,
        unit=U.liter,
    ),
    Compartment(
        sid="Vperi",
        name="peripheral compartment",
        value=1.0,
        sboTerm=SBO.PHYSICAL_COMPARTMENT,
        unit=U.liter,
    ),
    Compartment(
        sid="Vcent",
        name="central compartment",
        value=1.0,
        sboTerm=SBO.PHYSICAL_COMPARTMENT,
        unit=U.liter,
    ),
]

_m.species = [
    Species(
        sid="y_gut",
        name="y gut",
        compartment="Vgut",
        initialAmount=1.0,
        hasOnlySubstanceUnits=False,
        substanceUnit=U.mmole,
        sboTerm=SBO.SIMPLE_CHEMICAL,
        notes="""
        handled in amount not concentration
        """,
    ),
    Species(
        sid="y_cent",
        name="y central",
        compartment="Vcent",
        initialConcentration=0.0,
        substanceUnit=U.mmole,
        sboTerm=SBO.SIMPLE_CHEMICAL,
    ),
    Species(
        sid="y_peri",
        name="y peripheral",
        compartment="Vperi",
        initialConcentration=0.0,
        substanceUnit=U.mmole,
        sboTerm=SBO.SIMPLE_CHEMICAL,
    ),
]

_m.reactions = [
    Reaction(
        sid="ABSORPTION",
        name="absorption",
        equation="y_gut -> y_cent",
        formula="k_abs * y_gut",
        sboTerm=SBO.BIOCHEMICAL_REACTION,
        notes="""
        [mmole/min]
        absorption from gut
        """,
        pars=[
            Parameter(
                sid="k_abs",
                name="absorption rate",
                value=1.0,
                unit=U.l_per_min,
                sboTerm=SBO.KINETIC_CONSTANT,
            ),
        ],
    ),
    Reaction(
        sid="CLEARANCE",
        name="clearance",
        equation="y_cent ->",
        formula="CL * y_cent",
        sboTerm=SBO.BIOCHEMICAL_REACTION,
        notes="""
        [mmole/min]
        clearance from central compartment
        """,
        pars=[
            Parameter(
                sid="CL",
                name="clearance",
                value=1.0,
                unit=U.l_per_min,
                sboTerm=SBO.KINETIC_CONSTANT,
            ),
        ],
    ),
    Reaction(
        sid="R1",
        name="transport peripheral (R1)",
        equation="y_cent -> y_peri",
        formula="Q * y_cent",
        sboTerm=SBO.BIOCHEMICAL_REACTION,
        notes="""
        [mmole/min]
        distribution in peripheral compartment
        """,
        pars=[
            Parameter(
                sid="Q",
                name="distribution Q",
                value=1.0,
                unit=U.l_per_min,
                sboTerm=SBO.KINETIC_CONSTANT,
            ),
        ],
    ),
    Reaction(
        sid="R2",
        name="transport central (R2)",
        equation="y_peri -> y_cent",
        formula="Q * y_peri",
        sboTerm=SBO.BIOCHEMICAL_REACTION,
        notes="""
        [mmole/min]
        distribution from peripheral compartment
        """,
    ),
]


def create_simple_pk():
    from parvar import MODELS_DIR

    results: FactoryResult = create_model(
        model=_m,
        filepath=MODELS_DIR / f"{_m.sid}.xml",
        sbml_level=3,
        sbml_version=2,
        # validation_options=ValidationOptions(units_consistency=False)
    )

    # create differential equations
    md_path = MODELS_DIR / f"{_m.sid}.md"
    ode_factory = odefac.SBML2ODE.from_file(sbml_file=results.sbml_path)
    ode_factory.to_markdown(md_file=md_path)

    # visualize network
    visualize_sbml(sbml_path=results.sbml_path, delete_session=True)


if __name__ == "__main__":
    create_simple_pk()
