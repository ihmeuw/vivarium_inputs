import pytest
from unittest import mock

from ceam_inputs.data_artifact import ArtifactBuilder, split_entity_path, EntityError, _entities_by_type

def test_split_entity_path():
    assert split_entity_path("test.test") == ("test", "test")
    assert split_entity_path("type_only") == ("type_only", None)
    with pytest.raises(EntityError):
        split_entity_path("thing.thing.thing")


def test_entities_by_type():
    entities = ["test.test", "other.test", "other.other_test", "thing"]
    by_type = _entities_by_type(entities)
    assert set(by_type.keys()) == {"test", "other", "thing"}
    assert by_type["test"] == {"test"}
    assert by_type["other"] == {"test", "other_test"}
    assert by_type["thing"] == {None}


def test_entity_accumulation():
    builder = ArtifactBuilder()

    assert not builder.entities
    builder.data_container("test.test")
    assert len(builder.entities) == 1
    assert builder.entities == {"test.test"}

    builder.data_container("test.test2")
    assert builder.entities == {"test.test", "test.test2"}

    builder.data_container("type_only")
    assert builder.entities == {"test.test", "test.test2", "type_only"}


def test_request_and_dump(mocker):
    dump_mock = mocker.patch("ceam_inputs.data_artifact._dump")
    builder = ArtifactBuilder()
    builder.data_container("test.test")
    builder.data_container("test.other")
    builder.data_container("other_test")

    loaders = {"test": mock.Mock(), "other_test": mock.Mock()}

    builder.process("test.hdf", [1, 2], parallelism=1, loaders=loaders)

    assert loaders["test"].call_count == 2
    assert loaders["other_test"].call_count == 1

    # One call to dump because the mock loaders don't write anything out but the ArtifactBuilder
    # itself writes out a dimensions measure
    assert dump_mock.call_count == 1

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
    builder.save("/tmp/test_artifact.hdf", [180], parallelism=4)
