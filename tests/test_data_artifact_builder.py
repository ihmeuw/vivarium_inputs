import pytest
from unittest import mock

from ceam_inputs.data_artifact import ArtifactBuilder, EntityError, _entities_by_type

def test_entities_by_type():
    entities = ["test.test", "other.test.measure", "other.other_test.measure", "thing"]
    by_type = _entities_by_type(entities)
    assert set(by_type.keys()) == {"test", "other", "thing"}
    assert by_type["test"] == {"test"}
    assert by_type["other"] == {"test", "other_test"}
    assert by_type["thing"] == {None}


def test_entity_accumulation():
    builder = ArtifactBuilder()

    assert not builder.entities
    builder.data_container("test.test.measure")
    assert len(builder.entities) == 1
    assert builder.entities == {"test.test.measure"}

    builder.data_container("test.test2.other_measure")
    assert builder.entities == {"test.test.measure", "test.test2.other_measure"}

    builder.data_container("type_only")
    assert builder.entities == {"test.test.measure", "test.test2.other_measure", "type_only"}


def test_request_and_dump(mocker):
    dump_mock = mocker.patch("ceam_inputs.data_artifact._dump")
    builder = ArtifactBuilder()
    builder.data_container("test.test")
    builder.data_container("test.other")
    builder.data_container("other_test")

    loaders = {"test": mock.Mock(), "other_test": mock.Mock()}
    loaders["test"].return_value = [('a', None), ('b', None)]
    loaders["other_test"].return_value = [('a', None)]

    builder.process("test.hdf", [1, 2], parallelism=1, loaders=loaders)

    assert loaders["test"].call_count == 2
    assert loaders["other_test"].call_count == 1

    # NOTE Base just on the number of measures returned by the mock loaders and
    # the number of entities loaded you would expect this to be 5 but there is
    # one additional call to write out the full dimensions of the space the
    # artifact covers
    assert dump_mock.call_count == 6

@pytest.mark.skip(reason="Slow, really just example code")
def test_happy_path():
    builder = ArtifactBuilder()
    for cause in ["ischemic_heart_disease", "ischemic_stroke", "diarrheal_diseases"]:
        builder.data_container(f"cause.{cause}")
    for risk in ["high_systolic_blood_pressure", "unsafe_water_source"]:
        builder.data_container(f"risk_factor.{risk}")
    for sequela in ["acute_typhoid_infection", "mild_upper_respiratory_infections"]:
        builder.data_container(f"sequela.{sequela}")
    builder.data_container("risk_factor.correlations")
    builder.data_container("population")
    builder.data_container("healthcare_entity.outpatient_visits")
    for t in ["hypertension_drugs", "rota_vaccine", "hiv_positive_antiretroviral_treatment"]:
        builder.data_container(f"treatment_technology.{t}")
    for t in ["low_measles_vaccine_coverage_first_dose"]:
        builder.data_container(f"coverage_gap.{t}")
    for t in ["shigellosis",]:
        builder.data_container(f"etiology.{t}")
    for t in ["age_specific_fertility_rate", "live_births_by_sex", "dtp3_coverage_proportion"]:
        builder.data_container(f"covariate.{t}")
    builder.data_container(f"subregions")
    builder.process("/tmp/test_artifact.hdf", [180], parallelism=1)
